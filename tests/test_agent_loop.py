from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from repo_rivet.agent.controller import AgentController
from repo_rivet.agent.phases import WorkflowPhase
from repo_rivet.agent.runtime import AgentRuntimeState
from repo_rivet.agent.state import SessionState
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy
from repo_rivet.approval.models import ApprovalMode
from repo_rivet.editing.models import EditFileArguments
from repo_rivet.llm.base import ModelContextLengthError, ModelResponse, ModelStreamInterrupted
from repo_rivet.llm.openai_compatible import ModelRequestError
from repo_rivet.llm.parser import ResponseParseError
from repo_rivet.llm.protocol import validate_tool_call_protocol
from repo_rivet.memory.models import MemoryState, Message
from repo_rivet.memory.store import MemoryStore
from repo_rivet.planning.models import PlanArtifact, PlanStatus, WorkflowMode
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.tools.registry import ToolRegistry
from repo_rivet.verification.models import ModelErrorRecord
from tests.fakes import FakeModelClient, FakeToolRegistry


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def log(self, event_type: str, **data: object) -> None:
        self.events.append((event_type, data))


def call(call_id: str, name: str, arguments: dict) -> ToolCall:  # type: ignore[type-arg]
    return ToolCall(id=call_id, name=name, arguments=arguments)


def decision(call_id: str, next_tool: str, summary: str = "Use the declared tool.") -> ToolCall:
    return call(
        call_id,
        "record_decision",
        {
            "phase": "decision",
            "current_goal": "Complete the requested change safely",
            "summary": summary,
            "next_tool": next_tool,
            "next_tool_argument_summary": "bounded task action",
            "expected_result": "The action completes with a locally observed result",
            "confidence": 0.8,
        },
    )


def verification_plan(call_id: str = "verify-plan") -> ToolCall:
    return call(
        call_id,
        "register_verification",
        {
            "requirements": ["tests-pass"],
            "checks": [
                {
                    "check_id": "tests",
                    "title": "Run tests",
                    "kind": "test",
                    "command": {"program": "pytest", "args": ["-q"]},
                    "criteria": {"expected_exit_codes": [0]},
                    "required": True,
                    "claim_ids": ["tests-pass"],
                    "provenance": "model",
                }
            ],
        },
    )


def passed_verification_result(*, check_id: str = "tests", revision: int = 1) -> ToolResult:
    now = datetime.now(UTC).isoformat()
    return ToolResult(
        ok=True,
        output="passed",
        metadata={
            "exit_code": 0,
            "verification_result": {
                "check_id": check_id,
                "status": "passed",
                "workspace_revision": revision,
                "exit_code": 0,
                "reasons": ["all registered success criteria passed"],
                "started_at": now,
                "finished_at": now,
            },
        },
    )


def failed_verification_result(*, check_id: str = "tests", revision: int = 1) -> ToolResult:
    now = datetime.now(UTC).isoformat()
    return ToolResult(
        ok=True,
        output="tests failed",
        metadata={
            "exit_code": 1,
            "verification_result": {
                "check_id": check_id,
                "status": "failed",
                "workspace_revision": revision,
                "exit_code": 1,
                "reasons": ["exit code 1 was not one of [0]"],
                "started_at": now,
                "finished_at": now,
            },
        },
    )


def inconclusive_verification_result(
    *, check_id: str = "script_syntax", revision: int = 0
) -> ToolResult:
    now = datetime.now(UTC).isoformat()
    return ToolResult(
        ok=True,
        output="Command finished with exit code 0.",
        metadata={
            "exit_code": 0,
            "verification_result": {
                "check_id": check_id,
                "status": "inconclusive",
                "workspace_revision": revision,
                "exit_code": 0,
                "reasons": ["custom check has no output or artifact oracle"],
                "started_at": now,
                "finished_at": now,
            },
        },
    )


def script_syntax_plan(call_id: str, *, deterministic_oracle: bool) -> ToolCall:
    criteria: dict[str, object] = {"expected_exit_codes": [0]}
    if deterministic_oracle:
        criteria["stdout_exact"] = "ok"
    return call(
        call_id,
        "register_verification",
        {
            "requirements": ["script_syntax"],
            "checks": [
                {
                    "check_id": "script_syntax",
                    "title": "Check shell script syntax",
                    "kind": "custom",
                    "command": {"program": "bash", "args": ["-n", "script.sh"]},
                    "criteria": criteria,
                    "required": True,
                    "provenance": "model",
                }
            ],
        },
    )


def controller(
    model: FakeModelClient,
    tools: FakeToolRegistry,
    *,
    termination: TerminationPolicy | None = None,
    event_logger: RecordingSink | None = None,
) -> AgentController:
    return AgentController(
        model_client=model,
        tool_registry=cast(ToolRegistry, tools),
        termination_policy=termination,
        event_logger=event_logger,
    )


def test_agent_executes_tool_then_returns_final_text() -> None:
    read = call("1", "read_file", {"path": "app.py"})
    events = RecordingSink()
    model = FakeModelClient([ModelResponse(tool_calls=[read]), ModelResponse(content="Done.")])
    tools = FakeToolRegistry([ToolResult(ok=True, output="content")])

    result = controller(model, tools, event_logger=events).run("inspect the project")

    assert result.status == "success"
    assert result.summary == "Done."
    assert result.step_count == 2
    assert result.tool_call_count == 1
    session_end = next(data for name, data in events.events if name == "session_end")
    assert session_end["summary"] == "Done."


def test_events_distinguish_unexecuted_rejections_from_real_failures() -> None:
    rejected = call("write-rejected", "write_file", {"path": "app.py", "content": "x"})
    failed_read = call("read-failed", "read_file", {"path": "missing.py"})
    events = RecordingSink()
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[rejected]),
            ModelResponse(tool_calls=[failed_read]),
            ModelResponse(content="The requested operations did not change the workspace."),
        ]
    )
    tools = FakeToolRegistry([ToolResult(ok=False, output="", error="missing.py does not exist")])

    result = controller(model, tools, event_logger=events).run("inspect and update app.py")

    assert result.status == "success"
    rejected_result = next(
        data
        for name, data in events.events
        if name == "tool_result" and data.get("tool_call_id") == "write-rejected"
    )
    failed_observation = next(
        data
        for name, data in events.events
        if name == "observation" and data.get("tool_call_id") == "read-failed"
    )
    assert rejected_result["executed"] is False
    assert failed_observation["execution_attempted"] is True
    assert failed_observation["ok"] is False


def test_model_wait_state_is_not_persisted_as_session_status(tmp_path: Path) -> None:
    class RecordingMemoryStore(MemoryStore):
        def __init__(self, session_dir: Path) -> None:
            super().__init__(session_dir)
            self.saved_statuses: list[str] = []

        def save_state(self, memory: MemoryState, **runtime_state: object) -> None:
            self.saved_statuses.append(str(runtime_state.get("status", memory.status)))
            super().save_state(memory, **runtime_state)

    store = RecordingMemoryStore(tmp_path / "session")
    model = FakeModelClient([ModelResponse(content="Done.")])
    agent = AgentController(
        model_client=model,
        tool_registry=cast(ToolRegistry, FakeToolRegistry([])),
        memory_store=store,
    )

    result = agent.run("inspect the project")

    assert result.status == "success"
    assert "waiting" not in store.saved_statuses
    assert store.load_state().status == "completed"


def test_reasoning_effort_is_selected_by_task_phase_with_configured_ceiling() -> None:
    model = FakeModelClient([])
    model.reasoning_effort_ceiling = "high"  # type: ignore[attr-defined]
    agent = controller(model, FakeToolRegistry([]))

    routine = agent._model_request_options(SessionState(task="edit one file"))
    planning = agent._model_request_options(
        SessionState(task="design a refactor", workflow_mode=WorkflowMode.PLANNING)
    )
    recovery = agent._model_request_options(
        SessionState(task="repair a failure", verification_plan_recovery_attempts=1)
    )

    assert routine and routine.reasoning_effort == "low"
    assert planning and planning.reasoning_effort == "medium"
    assert recovery and recovery.reasoning_effort == "high"


def test_live_approval_mode_is_applied_at_controller_boundary() -> None:
    class MutableApprovalEngine:
        def __init__(self) -> None:
            self.mode = ApprovalMode.SAFE_AUTO

        def set_mode(self, mode: ApprovalMode) -> None:
            self.mode = mode

    tools = FakeToolRegistry([])
    tools.approval_engine = MutableApprovalEngine()  # type: ignore[attr-defined]
    agent = controller(FakeModelClient([]), tools)
    agent.set_runtime_settings_source(lambda: {"approval_mode": "llm-auto"})
    state = SessionState(task="continue")
    memory = MemoryState(session_id="live-settings")

    changed = agent._apply_runtime_settings(state=state, memory=memory)

    assert changed
    assert tools.approval_engine.mode == ApprovalMode.LLM_AUTO  # type: ignore[attr-defined]
    assert memory.approval_mode_override == ApprovalMode.LLM_AUTO


def test_long_final_response_is_bounded_only_in_assessment_memory() -> None:
    summary = "Detailed result. " + ("x" * 2_500)
    model = FakeModelClient([ModelResponse(content=summary)])
    memory = MemoryState(session_id="long-final-response")

    result = controller(model, FakeToolRegistry([])).run("inspect", memory=memory)

    assert result.status == "success"
    assert result.summary == summary
    assert memory.candidate_final_assessment is not None
    assert len(memory.candidate_final_assessment.summary) == 2_000
    assert memory.candidate_final_assessment.summary.endswith(
        "[Assessment summary truncated; full response remains in conversation history.]"
    )
    assert memory.messages[-1].content == summary


def test_empty_model_response_is_replaced_with_local_feedback() -> None:
    model = FakeModelClient([ModelResponse(), ModelResponse(content="Continued successfully.")])
    tools = FakeToolRegistry([])
    memory = MemoryState(session_id="empty-response")

    result = controller(model, tools).run("continue the task", memory=memory)

    assert result.status == "success"
    assert len(model.requests) == 2
    second_history = model.requests[1]["messages"]
    validate_tool_call_protocol(second_history)
    assert any(
        message.get("role") == "system" and "response was empty" in message.get("content", "")
        for message in second_history
    )
    assert all(
        message.get("role") != "assistant"
        or bool((message.get("content") or "").strip() or message.get("tool_calls"))
        for message in second_history
    )


def test_user_redirect_restarts_current_model_turn_without_ending_run() -> None:
    model = FakeModelClient(
        [
            ModelStreamInterrupted("redirected"),
            ModelResponse(content="Changed direction and completed."),
        ]
    )
    memory = MemoryState(session_id="redirected-run")
    events = RecordingSink()
    agent = controller(model, FakeToolRegistry([]), event_logger=events)
    source_calls = 0

    def steering_source() -> list[str]:
        nonlocal source_calls
        source_calls += 1
        return ["Use C++ instead."] if source_calls == 2 else []

    agent.set_steering_source(steering_source)
    result = agent.run("Use JavaScript.", memory=memory)

    assert result.status == "success"
    assert result.summary == "Changed direction and completed."
    assert [message.content for message in memory.messages if message.role == "user"] == [
        "Use JavaScript.",
        "Use C++ instead.",
    ]
    assert "Use C++ instead." in [
        message["content"] for message in model.requests[1]["messages"] if message["role"] == "user"
    ]
    assert any(name == "model_redirected" for name, _data in events.events)


def test_interrupted_tool_call_is_closed_before_same_process_continues(tmp_path: Path) -> None:
    read = call("interrupted-read", "read_file", {"path": "app.py"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[read]),
            ModelResponse(content="Continued after the interruption."),
        ]
    )

    class InterruptOnceRegistry(FakeToolRegistry):
        def execute(self, tool_call: ToolCall) -> ToolResult:
            self.calls.append(tool_call)
            raise KeyboardInterrupt

    tools = InterruptOnceRegistry([])
    events = RecordingSink()
    memory = MemoryState(session_id="same-process-interruption")
    agent = AgentController(
        model_client=model,
        tool_registry=cast(ToolRegistry, tools),
        event_logger=events,
        memory_store=MemoryStore(tmp_path / "session"),
    )

    interrupted = agent.run("inspect app.py", memory=memory)
    continued = agent.run("continue", memory=memory)

    assert interrupted.status == "stopped"
    assert interrupted.reason == "interrupted by user"
    assert continued.status == "success"
    validate_tool_call_protocol(model.requests[1]["messages"])
    synthetic = next(
        message
        for message in memory.messages
        if message.role == "tool" and message.tool_call_id == "interrupted-read"
    )
    assert "not retried" in (synthetic.content or "")
    assert any(event == "interrupted_history_repaired" for event, _ in events.events)


def test_resume_repairs_tool_result_that_was_appended_after_user_message() -> None:
    model = FakeModelClient([ModelResponse(content="Recovered the saved conversation.")])
    tools = FakeToolRegistry([])
    memory = MemoryState(
        session_id="misordered-tool-result",
        messages=[
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "write-1",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            ),
            Message(role="user", content="continue after interruption"),
            Message(role="tool", tool_call_id="write-1", content='{"ok": true}'),
        ],
    )

    result = controller(model, tools).run("continue safely", memory=memory)

    assert result.status == "success"
    validate_tool_call_protocol(model.requests[0]["messages"])
    roles = [message.role for message in memory.messages[:3]]
    assert roles == ["assistant", "tool", "user"]
    repaired_result = memory.messages[1]
    assert repaired_result.tool_call_id == "write-1"
    assert "interrupted_tool_call" in (repaired_result.content or "")


def test_length_response_without_visible_content_is_not_replayed() -> None:
    model = FakeModelClient(
        [
            ModelResponse(finish_reason="length"),
            ModelResponse(content="Recovered after truncation."),
        ]
    )
    tools = FakeToolRegistry([])

    result = controller(model, tools).run("continue after truncation")

    assert result.status == "success"
    validate_tool_call_protocol(model.requests[1]["messages"])
    assert all(
        message.get("role") != "assistant"
        or bool((message.get("content") or "").strip() or message.get("tool_calls"))
        for message in model.requests[1]["messages"]
    )
    assert model.requests[1]["options"].thinking_enabled is False


def test_length_response_without_reasoning_restarts_without_partial_content() -> None:
    model = FakeModelClient(
        [
            ModelResponse(content="unfinished private analysis", finish_reason="length"),
            ModelResponse(content="Recovered from durable facts."),
        ]
    )

    result = controller(model, FakeToolRegistry([])).run("continue safely")

    assert result.status == "success"
    second_messages = model.requests[1]["messages"]
    assert all(
        message.get("content") != "unfinished private analysis" for message in second_messages
    )
    assert model.requests[1]["options"].thinking_enabled is False
    assert "without returning replayable reasoning state" in second_messages[-1]["content"]


def test_agent_automatically_runs_registered_checks_before_finishing() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[verification_plan(), decision("d1", "write_file"), write]),
            ModelResponse(content="Implemented and ready for deterministic verification."),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=True, output="written"),
            passed_verification_result(),
        ]
    )

    result = controller(model, tools).run("fix the bug")

    assert result.status == "success"
    assert result.summary == "Implemented and ready for deterministic verification."
    assert result.modified_files == ("app.py",)
    assert result.verification_status.value == "passed"
    assert [item.name for item in tools.calls] == ["write_file", "run_verification"]
    assert len(model.requests) == 2


def test_failed_automatic_verification_requires_repair_before_same_revision_retry() -> None:
    write = call("write", "write_file", {"path": "app.py", "content": "new"})
    repeated_check = call("repeat", "run_verification", {"check_id": "tests"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[verification_plan(), decision("d1", "write_file"), write]),
            ModelResponse(content="Implementation is ready for verification."),
            ModelResponse(tool_calls=[repeated_check]),
            ModelResponse(content="The failing test requires a source repair first."),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=True, output="written"),
            failed_verification_result(),
        ]
    )
    events = RecordingSink()

    result = controller(model, tools, event_logger=events).run("fix the bug")

    assert result.status == "incomplete"
    assert [item.name for item in tools.calls] == ["write_file", "run_verification"]
    blocked = next(
        message
        for message in model.requests[3]["messages"]
        if message.get("role") == "tool" and message.get("tool_call_id") == "repeat"
    )
    assert "Do not rerun it before a relevant workspace change" in str(blocked.get("content"))
    assert any(
        event_type == "verification_repair_required" and data.get("check_id") == "tests"
        for event_type, data in events.events
    )
    assert any(
        "verification_failed" in str(message.get("content"))
        for message in model.requests[2]["messages"]
    )


def test_started_verification_runs_remaining_required_checks_without_model_round_trips() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    compile_check = call("verify-compile", "run_verification", {"check_id": "compile"})
    plan = call(
        "verify-plan",
        "register_verification",
        {
            "requirements": ["compile", "smoke"],
            "checks": [
                {
                    "check_id": "compile",
                    "title": "Compile application",
                    "kind": "build",
                    "command": {"program": "builder"},
                    "required": True,
                    "provenance": "model",
                },
                {
                    "check_id": "smoke",
                    "title": "Run smoke test",
                    "kind": "smoke",
                    "command": {"program": "app", "args": ["--test"]},
                    "required": True,
                    "provenance": "model",
                },
            ],
        },
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[plan, decision("d1", "write_file"), write]),
            ModelResponse(tool_calls=[compile_check]),
            ModelResponse(content="Implemented and all registered checks passed."),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=True, output="written"),
            passed_verification_result(check_id="compile"),
            passed_verification_result(check_id="smoke"),
        ]
    )

    event_logger = RecordingSink()
    result = controller(
        model,
        tools,
        termination=TerminationPolicy(TerminationConfig(max_steps=3)),
        event_logger=event_logger,
    ).run("implement and verify")

    assert result.status == "success"
    assert result.verification_status.value == "passed"
    assert [item.arguments.get("check_id") for item in tools.calls[1:]] == [
        "compile",
        "smoke",
    ]
    assert len(model.requests) == 3
    assert any(
        "verification_complete" in str(message.get("content"))
        for message in model.requests[2]["messages"]
    )
    assert model.requests[2]["tools"] == []
    assert any(
        event_type == "plan_authorized_action"
        and data.get("tool") == "run_verification"
        and data.get("requested_by") == "model"
        for event_type, data in event_logger.events
    )


def test_inconclusive_verification_requires_direct_complete_plan_revision() -> None:
    first_check = call("run-1", "run_verification", {"check_id": "script_syntax"})
    second_check = call("run-2", "run_verification", {"check_id": "script_syntax"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[script_syntax_plan("plan-1", deterministic_oracle=True)]),
            ModelResponse(tool_calls=[first_check]),
            ModelResponse(tool_calls=[decision("unneeded-decision", "register_verification")]),
            ModelResponse(tool_calls=[script_syntax_plan("plan-2", deterministic_oracle=True)]),
            ModelResponse(tool_calls=[second_check]),
            ModelResponse(content="Shell script syntax is valid."),
        ]
    )
    tools = FakeToolRegistry(
        [
            inconclusive_verification_result(),
            passed_verification_result(check_id="script_syntax", revision=0),
        ]
    )
    memory = MemoryState(session_id="inconclusive-verification-recovery")

    result = controller(
        model,
        tools,
    ).run("validate the scripts", memory=memory)

    assert result.status == "success"
    assert [executed.name for executed in tools.calls] == [
        "run_verification",
        "run_verification",
    ]
    rejected_decision = next(
        message
        for message in memory.messages
        if message.role == "tool" and message.tool_call_id == "unneeded-decision"
    )
    assert "verification_plan_revision_required" in (rejected_decision.content or "")
    revision_request = model.requests[2]["messages"]
    assert any(
        "does not require record_decision" in str(message.get("content"))
        for message in revision_request
    )
    assert memory.verification_plan_revision_required is False
    assert memory.verification_plan_revision_reason is None


def test_automatic_inconclusive_verification_enters_plan_revision_recovery() -> None:
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[script_syntax_plan("plan-1", deterministic_oracle=True)]),
            ModelResponse(content="The scripts are ready."),
            ModelResponse(tool_calls=[script_syntax_plan("plan-2", deterministic_oracle=True)]),
            ModelResponse(content="The scripts are ready."),
        ]
    )
    tools = FakeToolRegistry(
        [
            inconclusive_verification_result(),
            passed_verification_result(check_id="script_syntax", revision=0),
        ]
    )
    memory = MemoryState(session_id="automatic-inconclusive-verification-recovery")

    result = controller(model, tools).run("validate the scripts", memory=memory)

    assert result.status == "success"
    assert len(model.requests) == 4
    assert [executed.arguments["check_id"] for executed in tools.calls] == [
        "script_syntax",
        "script_syntax",
    ]
    assert any(
        "verification_plan_revision_required" in str(message.get("content"))
        for message in model.requests[2]["messages"]
    )
    assert memory.verification_plan_revision_required is False


def test_verification_plan_revision_recovery_is_bounded_for_repeated_final_text() -> None:
    model = FakeModelClient([ModelResponse(content="Already done.") for _item in range(3)])
    memory = MemoryState(session_id="bounded-verification-plan-revision")
    memory.verification_plan_revision_required = True
    memory.verification_plan_revision_reason = "behavior verification kind is missing"
    memory.verification_plan_revision_guidance = (
        "Call register_verification with a complete replacement plan."
    )

    result = controller(model, FakeToolRegistry([])).run("continue", memory=memory)

    assert result.status == "incomplete"
    assert result.reason is not None
    assert "revision was not provided after 3 recovery attempts" in result.reason
    assert len(model.requests) == 3


def test_verification_pass_on_final_model_step_finishes_instead_of_stopping() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    run_check = call("verify-tests", "run_verification", {"check_id": "tests"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[verification_plan(), decision("d1", "write_file"), write]),
            ModelResponse(tool_calls=[run_check]),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=True, output="written"),
            passed_verification_result(),
        ]
    )

    result = controller(
        model,
        tools,
        termination=TerminationPolicy(TerminationConfig(max_steps=2)),
    ).run("implement and verify")

    assert result.status == "success"
    assert result.step_count == 2
    assert result.verification_status.value == "passed"
    assert result.reason is None
    assert result.summary == (
        "Completed the requested task. All required verification checks passed. "
        "Modified files: app.py."
    )


def test_finalization_disables_tools_and_does_not_repeat_passed_check() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    first_check = call("verify-tests", "run_verification", {"check_id": "tests"})
    repeated_check = call("repeat-tests", "run_verification", {"check_id": "tests"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[verification_plan(), decision("d1", "write_file"), write]),
            ModelResponse(tool_calls=[first_check]),
            ModelResponse(tool_calls=[repeated_check]),
        ]
    )

    class NonEmptySchemaRegistry(FakeToolRegistry):
        def schemas(self) -> list[dict[str, object]]:
            return [
                {
                    "type": "function",
                    "function": {"name": "run_verification", "parameters": {}},
                }
            ]

    tools = NonEmptySchemaRegistry(
        [ToolResult(ok=True, output="written"), passed_verification_result()]
    )

    result = controller(model, tools).run("implement and verify")

    assert result.status == "success"
    assert result.verification_status.value == "passed"
    assert [item.name for item in tools.calls] == ["write_file", "run_verification"]
    assert model.requests[2]["tools"] == tools.schemas()
    assert model.requests[2]["options"].tool_choice == "none"


def test_finalization_discards_dsml_tool_markup_from_result_and_history() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    run_check = call("verify-tests", "run_verification", {"check_id": "tests"})
    leaked = (
        "Build passed. Now run the selftest.\n\n"
        '<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="record_decision">\n'
        '<｜｜DSML｜｜parameter name="phase">decision\\</｜｜DSML｜｜parameter>\n'
        "\\</｜｜DSML｜｜invoke>\n\\</｜｜DSML｜｜tool_calls>"
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[verification_plan(), decision("d1", "write_file"), write]),
            ModelResponse(tool_calls=[run_check]),
            ModelResponse(content=leaked),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=True, output="written"),
            passed_verification_result(),
        ]
    )
    memory = MemoryState(session_id="dsml-finalization")

    result = controller(model, tools).run("implement and verify", memory=memory)

    assert result.status == "success"
    assert result.summary == (
        "Completed the requested task. All required verification checks passed. "
        "Modified files: app.py."
    )
    assert "DSML" not in result.summary
    assert all("DSML" not in (message.content or "") for message in memory.messages)
    assert all(message.is_valid_provider_message() for message in memory.messages)
    validate_tool_call_protocol([message.as_chat_message() for message in memory.messages])


def test_controller_saves_paused_session_when_approval_stops_run(tmp_path: Path) -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    memory = MemoryState(session_id="verification-abort")

    class StatusInspectingRegistry(FakeToolRegistry):
        phase_at_verification: WorkflowPhase | None = None

        def execute(self, tool_call: ToolCall) -> ToolResult:
            if tool_call.name == "run_verification":
                self.phase_at_verification = (
                    memory.runtime.phase if memory.runtime is not None else None
                )
            return super().execute(tool_call)

    tools = StatusInspectingRegistry(
        [
            ToolResult(ok=True, output="written"),
            ToolResult(
                ok=False,
                output="",
                error="denied",
                error_code="approval_denied",
                metadata={"approval_abort": True},
            ),
        ]
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[verification_plan(), decision("d1", "write_file"), write]),
            ModelResponse(content="Ready for verification."),
        ]
    )
    store = MemoryStore(tmp_path / "session")

    result = AgentController(
        model_client=model,
        tool_registry=cast(ToolRegistry, tools),
        memory_store=store,
    ).run("fix", memory=memory)

    assert result.status == "stopped"
    assert tools.phase_at_verification == WorkflowPhase.EXECUTING_ACTION
    assert memory.verification_results["tests"].status.value == "error"
    assert memory.status == "paused"
    assert store.load_state().status == "paused"


def test_file_change_without_verification_plan_is_blocked_before_execution() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[decision("d1", "write_file")]),
            ModelResponse(tool_calls=[write]),
            ModelResponse(tool_calls=[verification_plan()]),
            ModelResponse(
                tool_calls=[call("2", "write_file", {"path": "app.py", "content": "new"})]
            ),
            ModelResponse(content="Implemented and verified."),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=True, output="written"),
            passed_verification_result(),
        ]
    )
    memory = MemoryState(session_id="missing-plan")
    events = RecordingSink()

    result = controller(model, tools, event_logger=events).run("fix the bug", memory=memory)

    assert result.status == "success"
    assert len(model.requests) == 5
    assert model.requests[1]["options"].required_tool == "register_verification"
    assert model.requests[2]["options"].required_tool == "register_verification"
    assert [executed.name for executed in tools.calls] == ["write_file", "run_verification"]
    blocked_result = next(
        message
        for message in memory.messages
        if message.role == "tool" and message.tool_call_id == "1"
    )
    assert "verification_plan_missing" in (blocked_result.content or "")
    assert memory.verification_plan_recovery_decision is None
    assert (
        sum(
            event_type == "verification_plan_recovery_started"
            for event_type, _data in events.events
        )
        == 2
    )


def test_pending_file_decision_registers_verification_before_requesting_the_edit() -> None:
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[decision("d1", "write_file")]),
            ModelResponse(tool_calls=[verification_plan()]),
            ModelResponse(
                tool_calls=[call("write", "write_file", {"path": "app.py", "content": "new"})]
            ),
            ModelResponse(content="Implemented and verified."),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=True, output="written"),
            passed_verification_result(),
        ]
    )
    memory = MemoryState(session_id="preflight-plan")

    result = controller(model, tools).run("fix the bug", memory=memory)

    assert result.status == "success"
    assert len(model.requests) == 4
    assert model.requests[1]["options"].required_tool == "register_verification"
    assert [executed.name for executed in tools.calls] == ["write_file", "run_verification"]
    assert not any(
        message.role == "tool"
        and message.tool_call_id == "write"
        and "verification_plan_missing" in (message.content or "")
        for message in memory.messages
    )


def test_resumed_missing_plan_recovery_allows_protocol_correction() -> None:
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[decision("d1", "edit_file")]),
            ModelResponse(tool_calls=[verification_plan()]),
            ModelResponse(content="Recovered and verified."),
        ]
    )
    tools = FakeToolRegistry([passed_verification_result(revision=0)])
    memory = MemoryState(session_id="resumed-missing-plan")
    memory.modified_files.add("app.py")
    memory.verification_plan_recovery_attempts = 1

    result = controller(model, tools).run("continue", memory=memory)

    assert result.status == "success"
    assert len(model.requests) == 3
    assert model.requests[0]["options"].required_tool == "register_verification"
    assert model.requests[1]["options"].required_tool == "register_verification"
    assert memory.verification_plan_recovery_attempts == 0
    rejected_decision = next(
        message
        for message in memory.messages
        if message.role == "tool" and message.tool_call_id == "d1"
    )
    assert '"retryable": true' in (rejected_decision.content or "")
    second_recovery = model.requests[1]["messages"]
    assert any(
        "register_verification only" in str(message.get("content")) for message in second_recovery
    )


def test_run_verification_without_plan_recovers_before_execution() -> None:
    run_before_plan = call("missing-plan-check", "run_verification", {"check_id": "tests"})
    run_after_plan = call("registered-check", "run_verification", {"check_id": "tests"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[run_before_plan]),
            ModelResponse(tool_calls=[verification_plan()]),
            ModelResponse(tool_calls=[run_after_plan]),
            ModelResponse(content="The registered verification passed."),
        ]
    )
    tools = FakeToolRegistry([passed_verification_result(check_id="tests", revision=0)])
    memory = MemoryState(session_id="verification-request-without-plan")
    events = RecordingSink()

    result = controller(
        model,
        tools,
        event_logger=events,
    ).run("verify the current project", memory=memory)

    assert result.status == "success"
    assert [executed.id for executed in tools.calls] == ["registered-check"]
    rejected = next(
        message
        for message in memory.messages
        if message.role == "tool" and message.tool_call_id == "missing-plan-check"
    )
    assert "verification_plan_missing" in (rejected.content or "")
    assert memory.verification_plan_recovery_attempts == 0
    assert any(
        event_type == "verification_plan_recovery_started"
        and data.get("requested_check_ids") == ["tests"]
        for event_type, data in events.events
    )


def test_executing_verify_step_requests_plan_registration_before_model_action() -> None:
    memory = MemoryState(session_id="planned-verification-without-runtime-plan")
    memory.workflow_mode = WorkflowMode.EXECUTE
    memory.plan_artifact = PlanArtifact.model_validate(
        {
            "goal": "Verify the current project",
            "evidence_refs": ["obs-project"],
            "steps": [
                {
                    "step_id": "verify-tests",
                    "title": "Run tests",
                    "intent": "Confirm project behavior",
                    "evidence_refs": ["obs-project"],
                    "operation": "verify",
                    "verification_ids": ["tests"],
                    "risk": "low",
                }
            ],
            "verification": [
                {
                    "check_id": "tests",
                    "title": "Run tests",
                    "success_criteria": "The test process exits successfully",
                }
            ],
            "plan_id": "plan-tests",
            "status": PlanStatus.EXECUTING,
            "workspace_revision": 0,
        }
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[verification_plan()]),
            ModelResponse(
                tool_calls=[
                    call("registered-plan-check", "run_verification", {"check_id": "tests"})
                ]
            ),
            ModelResponse(content="The planned verification passed."),
        ]
    )
    tools = FakeToolRegistry([passed_verification_result(check_id="tests", revision=0)])
    events = RecordingSink()

    result = controller(model, tools, event_logger=events).run(
        "continue the approved plan",
        memory=memory,
    )

    assert result.status == "success", result.reason
    assert [executed.id for executed in tools.calls] == ["registered-plan-check"]
    assert tools.calls[0].arguments == {"check_id": "tests"}
    assert any(
        "current approved plan step requires verification checks" in str(message.get("content"))
        for message in model.requests[0]["messages"]
    )
    assert any(
        event_type == "verification_plan_recovery_started"
        and data.get("requested_check_ids") == ["tests"]
        for event_type, data in events.events
    )


def test_missing_plan_recovery_is_bounded() -> None:
    model = FakeModelClient(
        [
            ModelResponse(content="Still not registering a plan."),
            ModelResponse(content="Still not registering a plan."),
            ModelResponse(content="Still not registering a plan."),
        ]
    )
    memory = MemoryState(session_id="bounded-missing-plan")
    memory.modified_files.add("app.py")
    memory.verification_plan_recovery_attempts = 1

    result = controller(model, FakeToolRegistry([])).run("continue", memory=memory)

    assert result.status == "incomplete"
    assert result.reason == (
        "no executable verification plan was registered after 3 recovery attempts"
    )
    assert len(model.requests) == 3
    assert memory.verification_plan_recovery_attempts == 3


def test_missing_plan_recovery_isolates_malformed_and_wrong_edit_responses() -> None:
    malformed_edit = ResponseParseError(
        "Tool call edit_file contains invalid JSON arguments: Extra data",
        code="invalid_tool_arguments_json",
        tool_name="edit_file",
        argument_chars=1_818,
    )
    repeated_edit = call(
        "repeated-edit",
        "edit_file",
        {"path": "app.py", "snapshot_id": "snapshot", "operations": []},
    )
    model = FakeModelClient(
        [
            malformed_edit,
            ModelResponse(tool_calls=[repeated_edit]),
            malformed_edit,
        ]
    )
    events = RecordingSink()
    memory = MemoryState(session_id="bounded-missing-plan-wrong-tool")
    memory.modified_files.add("app.py")
    memory.verification_plan_recovery_attempts = 1

    result = controller(
        model,
        FakeToolRegistry([]),
        event_logger=events,
    ).run("continue", memory=memory)

    assert result.status == "incomplete"
    assert result.reason == (
        "no executable verification plan was registered after 3 recovery attempts"
    )
    assert memory.verification_plan_recovery_attempts == 3
    assert all(
        request["options"].required_tool == "register_verification" for request in model.requests
    )
    invalid_responses = [data for name, data in events.events if name == "model_response_invalid"]
    assert invalid_responses
    assert all(data["recovery"] == "required_tool" for data in invalid_responses)
    assert not any(data.get("recovery") == "bounded_edit" for data in invalid_responses)


def test_provider_block_preserves_candidate_assessment_and_error_details() -> None:
    provider_error = ModelRequestError(
        ModelErrorRecord(
            error_type="BadRequestError",
            status_code=400,
            error_code="invalid_messages",
            request_id="request-1",
            message="invalid messages",
            retryable=False,
            attempt=1,
            max_attempts=4,
            message_count=5,
            message_roles=["system", "user"],
            pending_tool_call_ids=[],
            request_size_bytes=100,
        )
    )
    model = FakeModelClient(
        [
            ModelResponse(content="Candidate final answer."),
            provider_error,
        ]
    )
    memory = MemoryState(session_id="blocked-provider")
    memory.modified_files.add("app.py")

    result = controller(
        model,
        FakeToolRegistry([]),
    ).run("fix", memory=memory)

    assert result.status == "blocked"
    assert result.reason == (
        "model request failed: BadRequestError "
        "(status 400, code invalid_messages, request request-1)"
    )
    assert memory.candidate_final_assessment
    assert memory.candidate_final_assessment.summary == "Candidate final answer."
    assert memory.last_model_error and memory.last_model_error.status_code == 400
    assert result.verification_status.value == "not_run"


def test_length_limited_reasoning_is_replayed_then_removed_from_memory() -> None:
    model = FakeModelClient(
        [
            ModelResponse(
                reasoning_content="private provider continuation",
                finish_reason="length",
            ),
            ModelResponse(content="Completed after continuation."),
        ]
    )
    memory = MemoryState(session_id="length-continuation")

    result = controller(model, FakeToolRegistry([])).run("inspect", memory=memory)

    assert result.status == "success"
    assert len(model.requests) == 2
    second_messages = model.requests[1]["messages"]
    assert second_messages[-2]["reasoning_content"] == "private provider continuation"
    assert "truncated" in second_messages[-1]["content"]
    assert model.requests[1]["options"].reasoning_effort == "low"
    assert all(message.reasoning_content is None for message in memory.messages)
    assert all(not message.ephemeral for message in memory.messages)


def test_completed_provider_reasoning_is_retained_only_for_live_follow_up() -> None:
    model = FakeModelClient(
        [
            ModelResponse(
                content="First task completed.",
                reasoning_content="opaque provider continuation",
            ),
            ModelResponse(content="Follow-up completed."),
        ]
    )
    memory = MemoryState(session_id="live-provider-prefix")
    agent = controller(model, FakeToolRegistry([]))

    first = agent.run("inspect", memory=memory)
    second = agent.run("continue", memory=memory)

    assert first.status == "success"
    assert second.status == "success"
    assert any(
        message.get("reasoning_content") == "opaque provider continuation"
        for message in model.requests[1]["messages"]
    )
    assert all(
        "reasoning_content" not in message for message in memory.model_dump(mode="json")["messages"]
    )


def test_repeated_reasoning_length_exhaustion_disables_thinking_and_restarts_from_facts() -> None:
    model = FakeModelClient(
        [
            ModelResponse(reasoning_content="first hidden attempt", finish_reason="length"),
            ModelResponse(reasoning_content="second hidden attempt", finish_reason="length"),
            ModelResponse(content="Recovered without more hidden reasoning."),
        ]
    )
    memory = MemoryState(session_id="thinking-fallback")

    result = controller(model, FakeToolRegistry([])).run("inspect", memory=memory)

    assert result.status == "success"
    assert model.requests[1]["options"].reasoning_effort == "low"
    assert model.requests[2]["options"].thinking_enabled is False
    final_messages = model.requests[2]["messages"]
    assert all("reasoning_content" not in message for message in final_messages)
    assert "Thinking is disabled" in final_messages[-1]["content"]


def test_provider_thinking_recovery_restarts_context_for_following_tool_turns() -> None:
    read = call("recovered-read", "read_file", {"path": "app.py"})
    model = FakeModelClient(
        [
            ModelResponse(
                content="Inspect the file directly.",
                tool_calls=[read],
                provider_thinking_disabled=True,
                reasoning_context_restart_required=True,
            ),
            ModelResponse(content="Inspection completed."),
        ]
    )
    memory = MemoryState(session_id="same-run-thinking-recovery")

    result = controller(
        model,
        FakeToolRegistry([ToolResult(ok=True, output="file contents")]),
    ).run("inspect", memory=memory)

    assert result.status == "success"
    assert memory.provider_requires_reasoning_content is True
    assert len(model.requests) == 2
    assert model.requests[0]["options"] is None
    assert model.requests[1]["options"] is None
    checkpoint = next(
        message
        for message in model.requests[1]["messages"]
        if message.get("role") == "system"
        and "Provider protocol checkpoint" in str(message.get("content"))
    )
    assert '"tool":"read_file"' in checkpoint["content"]
    assert "file contents" in checkpoint["content"]
    assert model.requests[1]["messages"][-1]["role"] == "system"
    assert not any(message.get("role") == "tool" for message in model.requests[1]["messages"])
    validate_tool_call_protocol(model.requests[1]["messages"])


def test_completed_task_scope_does_not_leak_files_or_verification_to_next_task() -> None:
    memory = MemoryState(session_id="new-task-scope")
    memory.modified_files.add("old.cpp")
    memory.last_agent_outcome = "success"

    result = controller(
        FakeModelClient([ModelResponse(content="New task inspected.")]),
        FakeToolRegistry([]),
    ).run("inspect a different task", memory=memory)

    assert result.status == "success"
    assert result.modified_files == ()
    assert result.verification_status.value == "not_applicable"


def test_missing_durable_reasoning_for_tool_history_restarts_thinking_context() -> None:
    memory = MemoryState(
        session_id="reasoning-resume",
        provider_requires_reasoning_content=True,
        messages=[
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "old-call",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            ),
            Message(role="tool", tool_call_id="old-call", content="{}"),
        ],
    )
    model = FakeModelClient([ModelResponse(content="Resumed safely.")])

    result = controller(model, FakeToolRegistry([])).run("continue", memory=memory)

    assert result.status == "success"
    assert model.requests[0]["options"] is None
    assert not any(message.get("role") == "tool" for message in model.requests[0]["messages"])
    checkpoint = next(
        message
        for message in model.requests[0]["messages"]
        if message.get("role") == "system"
        and "Provider protocol checkpoint" in str(message.get("content"))
    )
    assert '"tool":"read_file"' in checkpoint["content"]
    assert model.requests[0]["messages"][-1]["role"] == "user"
    validate_tool_call_protocol(model.requests[0]["messages"])


def test_agent_reuses_repeated_identical_read_without_executing_again() -> None:
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[call("1", "read_file", {"path": "app.py"})]),
            ModelResponse(tool_calls=[call("2", "read_file", {"path": "app.py"})]),
            ModelResponse(tool_calls=[call("3", "read_file", {"path": "app.py"})]),
            ModelResponse(content="Inspection complete."),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(
                ok=True,
                output="content",
                metadata={
                    "path": "app.py",
                    "sha256": "content-sha",
                    "snapshot_id": "snapshot-app",
                },
            )
        ]
    )
    events = RecordingSink()

    result = controller(
        model,
        tools,
        event_logger=events,
    ).run("inspect")

    assert result.status == "success"
    assert len(tools.calls) == 1
    assert sum(name == "action_result_reused" for name, _ in events.events) == 2


def test_successful_command_is_not_executed_twice_at_same_workspace_revision() -> None:
    first = call(
        "command-1",
        "run_command",
        {"command": "make test", "cwd": ".", "timeout_seconds": 60},
    )
    duplicate = call(
        "command-2",
        "run_command",
        {"command": "make test", "cwd": ".", "timeout_seconds": 60},
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[decision("decision-1", "run_command"), first]),
            ModelResponse(tool_calls=[decision("decision-2", "run_command"), duplicate]),
            ModelResponse(content="The command succeeded; no further action is required."),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(
                ok=True,
                output="Command finished with exit code 0.",
                metadata={"exit_code": 0},
            )
        ]
    )
    events = RecordingSink()

    result = controller(model, tools, event_logger=events).run("run the requested check")

    assert result.status == "success"
    assert len(tools.calls) == 1
    assert not any(name == "action_blocked" for name, _ in events.events)
    assert not any(
        name == "tool_result" and data.get("error_code") == "duplicate_successful_command"
        for name, data in events.events
    )
    assert any(name == "action_result_reused" for name, _ in events.events)
    assert {schema["function"]["name"] for schema in model.requests[2]["tools"]} == set()


def test_stale_edit_snapshot_is_refreshed_without_consuming_repetition_budget() -> None:
    fresh_snapshot = "b" * 64
    stale = ToolResult(
        ok=False,
        output="",
        error="File changed after it was read; read it again before editing",
        error_code="stale_snapshot",
        retryable=True,
        metadata={
            "execution_attempted": False,
            "current_total_lines": 500,
        },
    )
    refreshed = ToolResult(
        ok=True,
        output="[src/Board.cpp#FRESH lines=50-349 total=500]\n50│ current line",
        metadata={
            "path": "src/Board.cpp",
            "sha256": "fresh-content-hash",
            "snapshot_id": fresh_snapshot,
            "snapshot_tag": "FRESH",
            "start_line": 50,
            "end_line": 349,
            "fully_visible_end_line": 349,
        },
    )
    tools = FakeToolRegistry([refreshed])
    agent = controller(FakeModelClient([]), tools)
    state = SessionState(
        task="update Board",
        runtime=AgentRuntimeState.create("stale-edit-recovery"),
    )
    memory = MemoryState(session_id="stale-edit-recovery")
    edit = call(
        "stale-edit",
        "edit_file",
        {
            "path": "src/Board.cpp",
            "snapshot_id": "a" * 64,
            "operations": [
                {
                    "op": "replace",
                    "start_line": 70,
                    "end_line": 72,
                    "new_lines": ["replacement"],
                }
            ],
        },
    )

    terminal = agent._record_action_result(edit, stale, state=state, memory=memory)
    recovery_terminal = agent._recover_edit_precondition(
        edit,
        stale,
        state=state,
        memory=memory,
    )

    assert terminal is None
    assert recovery_terminal is None
    assert len(tools.calls) == 1
    assert tools.calls[0].name == "read_file"
    assert tools.calls[0].arguments == {
        "path": "src/Board.cpp",
        "start_line": 50,
        "end_line": 349,
    }
    assert state.consecutive_failures == 0
    assert memory.current_snapshots["src/Board.cpp"] == fresh_snapshot
    recovery = next(
        message
        for message in reversed(state.messages)
        if message.get("role") == "system"
        and "edit_context_recovered" in str(message.get("content"))
    )
    assert fresh_snapshot in recovery["content"]
    assert "Record a fresh matching decision" in recovery["content"]


def test_unseen_edit_range_is_read_automatically_and_preserves_decision() -> None:
    snapshot_id = "a" * 64
    edit = call(
        "edit-unseen",
        "edit_file",
        {
            "path": "main.cpp",
            "snapshot_id": snapshot_id,
            "operations": [
                {
                    "op": "replace",
                    "start_line": 145,
                    "end_line": 191,
                    "new_lines": ["replacement"],
                }
            ],
        },
    )
    retry = call("edit-retry", "edit_file", edit.arguments)
    unseen = ToolResult(
        ok=False,
        output="",
        error="Target lines 145-191 were not shown; read that range first",
        error_code="unseen_range",
        retryable=True,
        metadata={
            "execution_attempted": False,
            "path": "main.cpp",
            "snapshot_id": snapshot_id,
            "required_start_line": 145,
            "required_end_line": 191,
        },
    )
    refreshed = ToolResult(
        ok=True,
        output="[main.cpp#AAAAAAAA lines=125-381 total=381]\n145│ current line",
        metadata={
            "path": "main.cpp",
            "snapshot_id": snapshot_id,
            "snapshot_tag": "AAAAAAAA",
            "start_line": 125,
            "end_line": 381,
            "fully_visible_end_line": 381,
        },
    )
    edited = ToolResult(
        ok=True,
        output="Edited main.cpp.",
        metadata={
            "path": "main.cpp",
            "new_snapshot_id": "b" * 64,
            "workspace_revision": 1,
        },
    )
    model = FakeModelClient(
        [
            ModelResponse(
                tool_calls=[
                    verification_plan(),
                    decision("decision-edit", "edit_file"),
                    edit,
                ]
            ),
            ModelResponse(tool_calls=[retry]),
            ModelResponse(content="The edit is complete; run the registered verification."),
            ModelResponse(content="Edit completed and verification passed."),
        ]
    )
    tools = FakeToolRegistry(
        [
            unseen,
            refreshed,
            edited,
            passed_verification_result(revision=1),
        ]
    )
    events = RecordingSink()

    result = controller(model, tools, event_logger=events).run("update main.cpp")

    assert result.status == "success"
    assert [item.name for item in tools.calls[:3]] == [
        "edit_file",
        "read_file",
        "edit_file",
    ]
    assert tools.calls[1].arguments == {
        "path": "main.cpp",
        "start_line": 125,
        "end_line": 424,
    }
    assert not any(
        name == "action_blocked" and data.get("error_code") == "decision_validation_failed"
        for name, data in events.events
    )
    assert any(name == "edit_context_recovery_finished" for name, _ in events.events)


def test_agent_renews_step_checkpoint_once_for_observed_progress() -> None:
    read = call("read", "read_file", {"path": "app.py"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[read]),
            ResponseParseError("invalid response 1"),
            ResponseParseError("invalid response 2"),
            ResponseParseError("invalid response 3"),
        ]
    )
    tools = FakeToolRegistry([ToolResult(ok=True, output="value = 1")])
    events = RecordingSink()
    termination = TerminationPolicy(TerminationConfig(max_steps=2, max_empty_model_responses=10))

    result = controller(
        model,
        tools,
        termination=termination,
        event_logger=events,
    ).run("inspect and continue")

    assert result.status == "stopped"
    assert result.step_count == 4
    assert "no meaningful progress" in (result.reason or "")
    renewals = [data for name, data in events.events if name == "agent_step_checkpoint_renewed"]
    assert renewals == [
        {
            "previous_checkpoint": 2,
            "next_checkpoint": 4,
            "progress_revision": 1,
        }
    ]


def test_agent_recovers_from_invalid_model_response() -> None:
    model = FakeModelClient(
        [ResponseParseError("bad tool JSON"), ModelResponse(content="Recovered.")]
    )
    tools = FakeToolRegistry([])

    result = controller(model, tools).run("task")

    assert result.status == "success"
    assert result.step_count == 2
    assert any(
        "bad tool JSON" in message.get("content", "") for message in model.requests[1]["messages"]
    )


def test_agent_replaces_truncated_whole_file_edit_with_bounded_recovery() -> None:
    parse_error = ResponseParseError(
        "Tool call edit_file contains invalid JSON arguments: Unterminated string",
        code="invalid_tool_arguments_json",
        tool_name="edit_file",
        argument_chars=27_952,
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[decision("decision", "edit_file")]),
            ModelResponse(tool_calls=[verification_plan()]),
            parse_error,
            ModelResponse(content="Stopped before applying the malformed edit."),
            ModelResponse(content="No malformed edit was applied; verification passed."),
        ]
    )
    events = RecordingSink()

    result = controller(
        model,
        FakeToolRegistry([passed_verification_result(revision=0)]),
        event_logger=events,
    ).run("refactor the file")

    assert result.status == "success"
    recovery_messages = model.requests[3]["messages"]
    recovery = next(
        str(message.get("content", ""))
        for message in reversed(recovery_messages)
        if message.get("role") == "system"
        and "whole-file payload" in str(message.get("content", ""))
    )
    assert "snapshot-bound edit_file operation" in recovery
    assert "previous oversized edit decision has been cancelled" in recovery
    invalid_event = next(data for name, data in events.events if name == "model_response_invalid")
    assert invalid_event["recovery"] == "bounded_edit"
    assert invalid_event["pending_decision_cancelled"] is True
    assert invalid_event["argument_chars"] == 27_952


def test_malformed_edit_recovery_keeps_schema_stable_and_enforces_bounded_read() -> None:
    class RecoverySchemaRegistry(FakeToolRegistry):
        def schemas(self) -> list[dict[str, object]]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file range.",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "description": "Edit a file.",
                        "parameters": EditFileArguments.model_json_schema(),
                    },
                },
            ]

    parse_error = ResponseParseError(
        "Tool call edit_file contains invalid JSON arguments: Expecting ',' delimiter",
        code="invalid_tool_arguments_json",
        tool_name="edit_file",
        argument_chars=2_178,
    )
    read = call(
        "read-recovery",
        "read_file",
        {"path": "app.py", "start_line": 1, "end_line": 20},
    )
    model = FakeModelClient(
        [parse_error, ModelResponse(tool_calls=[read]), ModelResponse(content="Recovered.")]
    )
    model.reasoning_effort_ceiling = "max"  # type: ignore[attr-defined]
    events = RecordingSink()
    agent = AgentController(
        model_client=model,
        tool_registry=cast(
            ToolRegistry,
            RecoverySchemaRegistry([ToolResult(ok=True, output="x")]),
        ),
        event_logger=events,
    )

    result = agent.run("edit app.py")

    assert result.status == "success"
    assert model.requests[1]["tools"] == model.requests[0]["tools"]
    assert model.requests[2]["tools"] == model.requests[0]["tools"]
    assert model.requests[1]["options"].reasoning_effort == "high"
    invalid = next(data for name, data in events.events if name == "model_response_invalid")
    assert invalid["requires_read"] is True
    assert invalid["recovery_attempt"] == 1


def test_agent_returns_error_after_model_client_retries_fail() -> None:
    model = FakeModelClient([RuntimeError("provider unavailable")])
    tools = FakeToolRegistry([])

    result = controller(model, tools).run("task")

    assert result.status == "error"
    assert result.reason == "model request failed: RuntimeError"


def test_agent_passes_prior_conversation_to_context() -> None:
    model = FakeModelClient(
        [ModelResponse(content="First result."), ModelResponse(content="Continued.")]
    )
    tools = FakeToolRegistry([])
    memory = MemoryState(session_id="test-session")

    controller(model, tools).run("first request", memory=memory)
    result = controller(model, tools).run("continue", memory=memory)

    assert result.status == "success"
    contents = [message.get("content") for message in model.requests[1]["messages"]]
    task_spec = next(str(content) for content in contents if "Original task" in str(content))
    assert "first request" in task_spec
    assert "continue" not in task_spec
    assert "continue" in contents
    assert any("First result." in str(content) for content in contents)


def test_agent_prefers_provider_usage_for_persistent_token_totals() -> None:
    model = FakeModelClient([ModelResponse(content="Done.", input_tokens=321, output_tokens=12)])
    tools = FakeToolRegistry([])
    memory = MemoryState(session_id="test-session")
    events = RecordingSink()

    result = controller(model, tools, event_logger=events).run("task", memory=memory)

    assert result.status == "success"
    assert memory.total_input_tokens == 321
    assert memory.total_output_tokens == 12
    usage = next(data for event, data in events.events if event == "model_usage")
    assert usage["actual_prompt_tokens"] == 321
    assert usage["completion_tokens"] == 12
    assert usage["raw_estimated_prompt_tokens"]


def test_agent_compacts_and_retries_after_provider_context_overflow() -> None:
    model = FakeModelClient(
        [
            ModelContextLengthError("too long"),
            ModelResponse(content="Recovered.", input_tokens=200, output_tokens=10),
        ]
    )
    tools = FakeToolRegistry([])
    memory = MemoryState(session_id="test-session")
    for index in range(20):
        memory.messages.append(Message(role="assistant", content=f"old {index}"))

    result = controller(model, tools).run("task", memory=memory)

    assert result.status == "success"
    assert len(model.requests) == 2
    assert memory.context_overflow_count == 1
    assert memory.compaction_count >= 1


def test_agent_stops_after_bounded_context_overflow_retries() -> None:
    model = FakeModelClient([ModelContextLengthError("too long")] * 3)
    tools = FakeToolRegistry([])
    memory = MemoryState(session_id="test-session")

    result = controller(model, tools).run("task", memory=memory)

    assert result.status == "stopped"
    assert "after 2 compression retries" in (result.reason or "")
    assert len(model.requests) == 3
    assert memory.context_overflow_count == 3
