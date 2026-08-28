import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from repo_rivet.approval.engine import ApprovalEngine
from repo_rivet.approval.grant_store import ApprovalGrantStore
from repo_rivet.approval.hard_policy import HardSafetyPolicy
from repo_rivet.approval.human_approver import (
    NonInteractiveHumanApprover,
    TerminalHumanApprover,
)
from repo_rivet.approval.llm_reviewer import OpenAIApprovalReviewer
from repo_rivet.approval.models import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalMode,
    ApprovalRequest,
    ApprovalScope,
    Capability,
    LLMReviewResult,
    RiskLevel,
)
from repo_rivet.approval.normalizer import RequestNormalizer
from repo_rivet.approval.risk_analyzer import RiskAnalyzer
from repo_rivet.config import ApiConfig
from repo_rivet.memory.models import MemoryState
from repo_rivet.tools.base import ToolCall
from repo_rivet.tools.registry import create_default_registry


class FakeHumanApprover:
    def __init__(
        self,
        *,
        action: ApprovalAction = ApprovalAction.ALLOW,
        scope: ApprovalScope = ApprovalScope.ONCE,
        guidance: str | None = None,
    ) -> None:
        self.action = action
        self.scope = scope
        self.guidance = guidance
        self.requests: list[ApprovalRequest] = []
        self.llm_reviews: list[LLMReviewResult | None] = []

    def ask(
        self,
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None = None,
    ) -> ApprovalDecision:
        self.requests.append(request)
        self.llm_reviews.append(llm_review)
        return ApprovalDecision(
            action=self.action,
            source="human",
            reason="test decision",
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
            scope=self.scope,
            guidance=self.guidance,
        )


class FakeReviewer:
    def __init__(self, result: LLMReviewResult | None) -> None:
        self.result = result
        self.requests: list[ApprovalRequest] = []

    def review(self, request: ApprovalRequest) -> LLMReviewResult | None:
        self.requests.append(request)
        return self.result


class TimeoutReviewer:
    def review(self, request: ApprovalRequest) -> LLMReviewResult | None:
        raise TimeoutError("review timed out")


class EventCollector:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def log(self, event_type: str, **data: object) -> None:
        self.events.append((event_type, data))


def create_engine(
    workspace: Path,
    *,
    mode: ApprovalMode,
    human: FakeHumanApprover | NonInteractiveHumanApprover | None = None,
    reviewer: FakeReviewer | None = None,
) -> tuple[ApprovalEngine, MemoryState]:
    memory = MemoryState(session_id="approval-test")
    engine = ApprovalEngine(
        mode=mode,
        normalizer=RequestNormalizer(workspace),
        risk_analyzer=RiskAnalyzer(),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=human or FakeHumanApprover(),
        llm_reviewer=reviewer,
    )
    return engine, memory


def test_allow_all_approves_workspace_write_but_denies_escape(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    registry = create_default_registry(tmp_path, approval_engine=engine)

    allowed = registry.execute(
        ToolCall(
            id="write-1",
            name="write_file",
            arguments={"path": "inside.txt", "content": "ok"},
        )
    )
    denied = registry.execute(
        ToolCall(
            id="write-2",
            name="write_file",
            arguments={"path": "../outside.txt", "content": "no"},
        )
    )

    assert allowed.ok
    assert (tmp_path / "inside.txt").read_text(encoding="utf-8") == "ok"
    assert not denied.ok
    assert denied.error_code == "hard_policy_denied"
    assert not (tmp_path.parent / "outside.txt").exists()


def test_safe_auto_allows_normal_read_and_denies_sensitive_read(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    registry = create_default_registry(tmp_path, approval_engine=engine)

    normal = registry.execute(
        ToolCall(id="read-1", name="read_file", arguments={"path": "main.py"})
    )
    sensitive = registry.execute(
        ToolCall(id="read-2", name="read_file", arguments={"path": ".env"})
    )

    assert normal.ok
    assert not sensitive.ok
    assert sensitive.error_code == "hard_policy_denied"
    assert human.requests == []


def test_edit_approval_contains_preflight_diff_and_version_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "main.py"
    path.write_text("value = 1\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    events = EventCollector()
    engine.event_logger = events
    registry = create_default_registry(
        tmp_path,
        approval_engine=engine,
        event_logger=events,
    )
    read = registry.execute(
        ToolCall(id="read-edit", name="read_file", arguments={"path": "main.py"})
    )
    assert read.ok and read.metadata

    edited = registry.execute(
        ToolCall(
            id="edit-1",
            name="edit_file",
            arguments={
                "path": "main.py",
                "snapshot_id": read.metadata["snapshot_id"],
                "operations": [
                    {
                        "op": "replace",
                        "start_line": 1,
                        "end_line": 1,
                        "new_lines": ["value = 2"],
                    }
                ],
            },
        )
    )

    assert edited.ok and edited.metadata
    request = human.requests[-1]
    normalized = request.normalized_arguments
    assert request.tool_name == "edit_file"
    assert normalized["snapshot_id"] == read.metadata["snapshot_id"]
    assert normalized["prepared_live_hash"] == read.metadata["raw_bytes_hash"]
    assert "-value = 1" in normalized["diff_preview"]
    assert "+value = 2" in normalized["diff_preview"]
    assert normalized["operations"][0]["new_line_count"] == 1
    assert "new_lines" not in normalized["operations"][0]
    assert edited.metadata["workspace_revision"] == 1
    assert any(event == "edit_approved" for event, _ in events.events)


def test_large_line_deletion_is_classified_as_high_risk(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)

    outcome = engine.authorize(
        tool_name="edit_file",
        arguments={
            "path": "main.py",
            "snapshot_id": "a" * 64,
            "prepared_live_hash": "b" * 64,
            "operations": [{"op": "delete", "start_line": 1, "end_line": 100}],
            "diff_preview": "bounded preview",
        },
        capabilities={Capability.FILESYSTEM_READ, Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    )

    assert outcome.request.assessment.level == RiskLevel.HIGH
    assert any(
        "removes at least 100 lines" in reason for reason in outcome.request.assessment.reasons
    )


def test_always_ask_prompts_even_for_list_files(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALWAYS_ASK, human=human)
    registry = create_default_registry(tmp_path, approval_engine=engine)

    result = registry.execute(ToolCall(id="list-1", name="list_files", arguments={}))

    assert result.ok
    assert [request.tool_name for request in human.requests] == ["list_files"]


def test_verification_approval_reviews_the_registered_concrete_command(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    registry = create_default_registry(tmp_path, approval_engine=engine)
    assert registry.verification_runtime is not None
    registry.verification_runtime.bind(memory)
    registry.verification_runtime.register_plan(
        {
            "checks": [
                {
                    "check_id": "smoke",
                    "title": "Run smoke check",
                    "kind": "smoke",
                    "command": {"program": sys.executable, "args": ["-c", "print('ok')"]},
                    "criteria": {"expected_exit_codes": [0]},
                    "required": True,
                    "provenance": "model",
                }
            ]
        }
    )

    result = registry.execute(
        ToolCall(id="verify-1", name="run_verification", arguments={"check_id": "smoke"})
    )

    assert result.ok
    assert len(human.requests) == 1
    request = human.requests[0]
    assert request.tool_name == "run_verification"
    command = request.normalized_arguments["command"]
    assert command["program"] == sys.executable
    assert command["args"] == ["-c", "print('ok')"]


def test_llm_auto_accepts_high_confidence_medium_risk_review(tmp_path: Path) -> None:
    human = FakeHumanApprover(action=ApprovalAction.DENY)
    reviewer = FakeReviewer(
        LLMReviewResult(
            decision="allow",
            risk_level=RiskLevel.MEDIUM,
            confidence=0.95,
            reason="bounded test command",
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.action == ApprovalAction.ALLOW
    assert outcome.decision.source == "llm_reviewer"
    assert len(reviewer.requests) == 1
    assert human.requests == []


def test_invalid_or_unavailable_llm_review_falls_back_to_human(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    reviewer = FakeReviewer(None)
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.source == "human"
    assert len(human.requests) == 1


def test_llm_timeout_falls_back_to_human(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    memory = MemoryState(session_id="approval-test")
    engine = ApprovalEngine(
        mode=ApprovalMode.LLM_AUTO,
        normalizer=RequestNormalizer(tmp_path),
        risk_analyzer=RiskAnalyzer(),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=human,
        llm_reviewer=TimeoutReviewer(),
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.source == "human"
    assert len(human.requests) == 1


def test_openai_reviewer_treats_invalid_json_as_failure() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))]
    )
    completions = SimpleNamespace(create=lambda **_: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    reviewer = OpenAIApprovalReviewer(
        ApiConfig(
            api_key="test-secret",
            base_url="https://example.com/v1",
            model="reviewer",
            context_window_tokens=8_192,
        ),
        client=client,
    )
    request = ApprovalRequest(
        request_id="request",
        session_id="session",
        tool_name="run_command",
        arguments={},
        normalized_arguments={},
        workspace="/workspace",
        fingerprint="fingerprint",
        assessment={"level": RiskLevel.MEDIUM},
    )

    assert reviewer.review(request) is None


def test_llm_cannot_auto_approve_high_risk_network_command(tmp_path: Path) -> None:
    human = FakeHumanApprover(action=ApprovalAction.DENY)
    reviewer = FakeReviewer(
        LLMReviewResult(
            decision="allow",
            risk_level=RiskLevel.LOW,
            confidence=1,
            reason="unsafe optimistic review",
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "curl https://example.com", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.assessment.level == RiskLevel.HIGH
    assert outcome.decision.action == ApprovalAction.DENY
    assert reviewer.requests == []
    assert len(human.requests) == 1


def test_shell_syntax_is_never_obviously_safe(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest && echo done", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.assessment.level == RiskLevel.HIGH
    assert not outcome.request.assessment.obviously_safe

    registry = create_default_registry(tmp_path, approval_engine=engine)
    rejected = registry.execute(
        ToolCall(
            id="shell-1",
            name="run_command",
            arguments={"command": "pytest && echo done"},
        )
    )
    assert rejected.error_code == "invalid_command"


def test_session_grant_matches_only_exact_normalized_request(tmp_path: Path) -> None:
    human = FakeHumanApprover(scope=ApprovalScope.SESSION_EXACT)
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    common = {
        "tool_name": "write_file",
        "capabilities": {Capability.FILESYSTEM_WRITE},
        "session_id": engine.session_id,
    }

    first = engine.authorize(arguments={"path": "a.py", "content": "one"}, **common)
    repeated = engine.authorize(arguments={"path": "a.py", "content": "one"}, **common)
    changed = engine.authorize(arguments={"path": "a.py", "content": "two"}, **common)

    assert first.decision.source == "human"
    assert repeated.decision.source == "session_grant"
    assert changed.decision.source == "human"
    assert len(human.requests) == 2
    assert len(memory.approval_session_grants) == 2


def test_read_only_mode_allows_typed_reads_and_denies_writes_and_commands(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.READ_ONLY)
    registry = create_default_registry(tmp_path, approval_engine=engine)

    read = registry.execute(ToolCall(id="read-1", name="read_file", arguments={"path": "main.py"}))
    write = registry.execute(
        ToolCall(
            id="write-1",
            name="write_file",
            arguments={"path": "new.py", "content": "value = 2\n"},
        )
    )
    command = registry.execute(
        ToolCall(id="command-1", name="run_command", arguments={"command": "pwd"})
    )

    assert read.ok
    assert not write.ok
    assert write.error_code == "approval_denied"
    assert write.metadata and write.metadata["approval_source"] == "read_only_mode"
    assert not command.ok
    assert command.metadata and command.metadata["approval_source"] == "read_only_mode"
    assert not (tmp_path / "new.py").exists()


def test_read_only_mode_cannot_be_bypassed_by_prior_session_grant(tmp_path: Path) -> None:
    human = FakeHumanApprover(scope=ApprovalScope.SESSION_EXACT)
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    request = {
        "tool_name": "write_file",
        "arguments": {"path": "file.txt", "content": "value"},
        "capabilities": {Capability.FILESYSTEM_WRITE},
        "session_id": engine.session_id,
    }
    granted = engine.authorize(**request)

    engine.set_mode(ApprovalMode.READ_ONLY)
    denied = engine.authorize(**request)

    assert granted.decision.scope == ApprovalScope.SESSION_EXACT
    assert denied.decision.action == ApprovalAction.DENY
    assert denied.decision.source == "read_only_mode"
    assert memory.approval_mode_override == ApprovalMode.READ_ONLY


def test_switch_to_read_only_invalidates_pending_write_approval(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    approved = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    )

    engine.set_mode(ApprovalMode.READ_ONLY)
    revalidated = engine.revalidate(approved)

    assert revalidated is not None
    assert revalidated.action == ApprovalAction.DENY
    assert revalidated.source == "read_only_mode"


def test_mode_switch_updates_fixed_model_safety_rule(tmp_path: Path) -> None:
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO)
    memory.start_task(
        task="inspect project",
        workspace=str(tmp_path),
        system_prompt="system",
        safety_rules=["stay in workspace"],
        completion_rules=["report result"],
        max_steps=10,
    )

    engine.set_mode(ApprovalMode.READ_ONLY)

    assert memory.fixed is not None
    approval_rules = [
        rule for rule in memory.fixed.safety_rules if rule.startswith("Current approval mode:")
    ]
    assert approval_rules == [
        "Current approval mode: read-only. Only typed, workspace-confined file inspection "
        "tools are permitted; do not request writes or commands."
    ]


def test_symlink_change_invalidates_approval_before_execution(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    outcome = engine.authorize(
        tool_name="write_file",
        arguments={"path": "target/file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    )

    target.rmdir()
    target.symlink_to(outside, target_is_directory=True)
    stale = engine.revalidate(outcome)

    assert stale is not None
    assert stale.source == "execution_revalidation"
    assert stale.action == ApprovalAction.DENY


def test_noninteractive_safe_auto_denies_request_needing_human(tmp_path: Path) -> None:
    engine, memory = create_engine(
        tmp_path,
        mode=ApprovalMode.SAFE_AUTO,
        human=NonInteractiveHumanApprover(),
    )

    outcome = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    )

    assert outcome.decision.action == ApprovalAction.DENY
    assert outcome.decision.source == "non_interactive_policy"
    assert outcome.request.fingerprint in memory.denied_request_fingerprints


def test_repeated_once_denial_is_not_prompted_again_during_task(tmp_path: Path) -> None:
    human = FakeHumanApprover(action=ApprovalAction.DENY)
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    request = {
        "tool_name": "write_file",
        "arguments": {"path": "file.txt", "content": "value"},
        "capabilities": {Capability.FILESYSTEM_WRITE},
        "session_id": engine.session_id,
    }

    first = engine.authorize(**request)
    repeated = engine.authorize(**request)

    assert first.decision.source == "human"
    assert repeated.decision.source == "prior_denial"
    assert len(human.requests) == 1


def test_terminal_approval_uses_numbered_options_and_reprompts_invalid_choice(
    tmp_path: Path,
) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    request = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    ).request
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=160)
    choices = iter(["y", "2"])

    decision = TerminalHumanApprover(console, reader=lambda _: next(choices)).ask(request)

    output = buffer.getvalue()
    assert decision.action == ApprovalAction.ALLOW
    assert decision.scope == ApprovalScope.SESSION_EXACT
    assert "1  Approve once" in output
    assert "2  Approve this exact request for the session" in output
    assert "5  Abort agent" in output
    assert "Enter a number from 1 to 5" in output


def test_terminal_option_five_marks_agent_abort(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    request = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    ).request
    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    decision = TerminalHumanApprover(console, reader=lambda _: "5").ask(request)

    assert decision.action == ApprovalAction.DENY
    assert decision.abort_agent


def test_terminal_denial_can_include_direction_for_agent(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    request = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    ).request
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=180)
    answers = iter(["3", "先读取现有文件，只修改目标函数"])

    decision = TerminalHumanApprover(console, reader=lambda _: next(answers)).ask(request)

    assert decision.action == ApprovalAction.DENY
    assert decision.scope == ApprovalScope.ONCE
    assert decision.guidance == "先读取现有文件，只修改目标函数"
    assert "Deny once, with optional direction" in buffer.getvalue()


def test_guided_denial_is_returned_to_model_and_reused_for_exact_retry(tmp_path: Path) -> None:
    guidance = "Use edit_file on app.py instead of overwriting the file"
    human = FakeHumanApprover(action=ApprovalAction.DENY, guidance=guidance)
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    events = EventCollector()
    engine.event_logger = events
    registry = create_default_registry(tmp_path, approval_engine=engine)
    call = ToolCall(
        id="write-1",
        name="write_file",
        arguments={"path": "app.py", "content": "replacement"},
    )

    first = registry.execute(call)
    repeated = registry.execute(call)

    assert not first.ok and not repeated.ok
    assert f"User direction: {guidance}" in (first.error or "")
    assert f"User direction: {guidance}" in (repeated.error or "")
    assert first.metadata and first.metadata["approval_guidance"] == guidance
    assert len(human.requests) == 1
    assert list(memory.approval_denial_guidance.values()) == [guidance]
    decision_events = [data for event, data in events.events if event == "approval_decided"]
    assert decision_events[0]["guidance"] == guidance


def test_approval_events_cover_request_decision_and_execution(tmp_path: Path) -> None:
    memory = MemoryState(session_id="approval-test")
    events = EventCollector()
    engine = ApprovalEngine(
        mode=ApprovalMode.ALLOW_ALL,
        normalizer=RequestNormalizer(tmp_path),
        risk_analyzer=RiskAnalyzer(),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=FakeHumanApprover(),
        event_logger=events,
    )
    registry = create_default_registry(tmp_path, approval_engine=engine)

    result = registry.execute(ToolCall(id="list-1", name="list_files", arguments={}))

    assert result.ok
    assert [event[0] for event in events.events] == [
        "approval_requested",
        "approval_decided",
        "approved_tool_started",
        "approved_tool_executed",
    ]
