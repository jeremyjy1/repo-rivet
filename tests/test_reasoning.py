from pathlib import Path
from typing import cast

import pytest

from repo_rivet.agent.controller import AgentController
from repo_rivet.approval.engine import ApprovalEngine
from repo_rivet.approval.grant_store import ApprovalGrantStore
from repo_rivet.approval.hard_policy import HardSafetyPolicy
from repo_rivet.approval.human_approver import NonInteractiveHumanApprover
from repo_rivet.approval.models import ApprovalMode
from repo_rivet.approval.normalizer import RequestNormalizer
from repo_rivet.approval.risk_analyzer import RiskAnalyzer
from repo_rivet.llm.base import ModelResponse
from repo_rivet.llm.protocol import validate_tool_call_protocol
from repo_rivet.memory.models import MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.reasoning.manager import ReasoningManager
from repo_rivet.reasoning.models import ReasoningConfig, ReasoningPhase
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.tools.registry import ToolRegistry, create_default_registry
from tests.fakes import FakeModelClient, FakeToolRegistry


def decision(
    call_id: str,
    next_tool: str,
    *,
    phase: str = "decision",
    summary: str = "Use the declared action based on observed workspace state.",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name="record_decision",
        arguments={
            "phase": phase,
            "current_goal": "Complete the bounded task",
            "summary": summary,
            "evidence_refs": ["obs-prior"],
            "assumptions": [],
            "open_questions": [],
            "next_tool": next_tool,
            "next_tool_argument_summary": "target app.py",
            "expected_result": "The declared action produces a local observation",
            "confidence": 0.85,
        },
    )


def controller(
    responses: list[ModelResponse],
    tools: FakeToolRegistry,
    *,
    memory: MemoryState | None = None,
    reasoning_manager: ReasoningManager | None = None,
):  # type: ignore[no-untyped-def]
    state = memory or MemoryState(session_id="reasoning-test")
    agent = AgentController(
        model_client=FakeModelClient(responses),
        tool_registry=cast(ToolRegistry, tools),
        reasoning_manager=reasoning_manager,
    )
    return agent, state


def test_mutating_tool_without_decision_is_rejected_before_executor() -> None:
    write = ToolCall(id="write-1", name="write_file", arguments={"path": "app.py"})
    tools = FakeToolRegistry([])
    agent, memory = controller(
        [ModelResponse(tool_calls=[write]), ModelResponse(content="No change was made.")],
        tools,
    )

    result = agent.run("change app.py", memory=memory)

    assert result.status == "success"
    assert tools.calls == []
    assert memory.observation_events == []
    assert not memory.reflection_required
    tool_payload = next(
        message.content or "" for message in memory.messages if message.tool_call_id == "write-1"
    )
    assert "decision_validation_failed" in tool_payload


def test_declared_tool_mismatch_is_rejected_without_guessing_intent() -> None:
    write = ToolCall(id="write-1", name="edit_file", arguments={"path": "app.py"})
    declared_read = decision("decision-1", "read_file")
    tools = FakeToolRegistry([])
    agent, memory = controller(
        [
            ModelResponse(tool_calls=[declared_read, write]),
            ModelResponse(content="Stopped after the mismatch."),
        ],
        tools,
    )

    agent.run("change app.py", memory=memory)

    assert tools.calls == []
    assert memory.reasoning_events[-1].next_action
    assert memory.reasoning_events[-1].next_action.tool_name == "read_file"
    assert "actual tools were edit_file" in (memory.messages[-2].content or "")


def test_read_decision_cannot_cover_mutation_in_same_turn() -> None:
    read = ToolCall(id="read-1", name="read_file", arguments={"path": "README.txt"})
    write = ToolCall(
        id="write-1",
        name="write_file",
        arguments={"path": "app.py", "content": "new"},
    )
    tools = FakeToolRegistry([])
    agent, memory = controller(
        [
            ModelResponse(tool_calls=[decision("decision-1", "read_file"), read, write]),
            ModelResponse(content="Stopped after validation."),
        ],
        tools,
    )

    agent.run("inspect and change app.py", memory=memory)

    assert tools.calls == []
    assert memory.observation_events == []
    tool_payloads = [
        message.content or ""
        for message in memory.messages
        if message.tool_call_id in {"read-1", "write-1"}
    ]
    assert all("actual tools were write_file" in payload for payload in tool_payloads)


def test_matching_decision_executes_action_and_creates_executor_observation() -> None:
    write = ToolCall(
        id="write-1",
        name="write_file",
        arguments={"path": "app.py", "content": "new"},
    )
    tools = FakeToolRegistry([ToolResult(ok=True, output="written", metadata={"path": "app.py"})])
    agent, memory = controller(
        [
            ModelResponse(tool_calls=[decision("decision-1", "write_file"), write]),
            ModelResponse(content="Changed app.py."),
        ],
        tools,
    )

    agent.run("change app.py", memory=memory)

    assert tools.calls == [write]
    observation = memory.observation_events[-1]
    assert observation.ok
    assert observation.tool_call_id == "write-1"
    assert observation.affected_paths == ["app.py"]
    assert observation.event_id.startswith("obs-")
    tool_payload = next(
        message.content or "" for message in memory.messages if message.tool_call_id == "write-1"
    )
    assert observation.event_id in tool_payload


def test_more_than_one_state_changing_action_in_a_turn_is_rejected() -> None:
    first = ToolCall(id="write-1", name="write_file", arguments={"path": "a.py"})
    second = ToolCall(id="write-2", name="edit_file", arguments={"path": "b.py"})
    tools = FakeToolRegistry([])
    agent, memory = controller(
        [
            ModelResponse(tool_calls=[decision("decision-1", "write_file"), first, second]),
            ModelResponse(content="No actions executed."),
        ],
        tools,
    )

    agent.run("change files", memory=memory)

    assert tools.calls == []
    assert memory.observation_events == []
    assert not memory.reflection_required


def test_standalone_decision_authorizes_immediately_following_matching_action() -> None:
    command = ToolCall(
        id="command-1",
        name="run_command",
        arguments={"command": "python -m pytest -q"},
    )
    tools = FakeToolRegistry([ToolResult(ok=True, output="passed", metadata={"exit_code": 0})])
    agent, memory = controller(
        [
            ModelResponse(tool_calls=[decision("decision-1", "run_command")]),
            ModelResponse(tool_calls=[command]),
            ModelResponse(content="Verification passed."),
        ],
        tools,
    )

    result = agent.run("verify", memory=memory)

    assert result.status == "success"
    assert result.tool_call_count == 1
    assert tools.calls == [command]
    assert memory.observation_events[-1].tool_call_id == "command-1"


def test_protocol_rejection_does_not_require_reflection_or_count_as_tool_failure() -> None:
    rejected = ToolCall(
        id="command-rejected",
        name="run_command",
        arguments={"command": "python -m pytest -q"},
    )
    accepted = ToolCall(
        id="command-accepted",
        name="run_command",
        arguments={"command": "python -m pytest -q"},
    )
    tools = FakeToolRegistry([ToolResult(ok=True, output="passed", metadata={"exit_code": 0})])
    agent, memory = controller(
        [
            ModelResponse(tool_calls=[rejected]),
            ModelResponse(tool_calls=[decision("decision-1", "run_command")]),
            ModelResponse(tool_calls=[accepted]),
            ModelResponse(content="Verification passed."),
        ],
        tools,
    )

    result = agent.run("verify", memory=memory)

    assert result.status == "success"
    assert result.tool_call_count == 1
    assert tools.calls == [accepted]
    assert not memory.reflection_required
    assert all(event.tool_call_id != "command-rejected" for event in memory.observation_events)


def test_cross_turn_decision_is_one_shot_and_must_match() -> None:
    mismatched = ToolCall(
        id="write-mismatch",
        name="write_file",
        arguments={"path": "app.py", "content": "new"},
    )
    accepted = ToolCall(
        id="command-accepted",
        name="run_command",
        arguments={"command": "python -m pytest -q"},
    )
    tools = FakeToolRegistry([ToolResult(ok=True, output="passed", metadata={"exit_code": 0})])
    agent, memory = controller(
        [
            ModelResponse(tool_calls=[decision("decision-1", "run_command")]),
            ModelResponse(tool_calls=[mismatched]),
            ModelResponse(tool_calls=[decision("decision-2", "run_command")]),
            ModelResponse(tool_calls=[accepted]),
            ModelResponse(content="Verification passed."),
        ],
        tools,
    )

    result = agent.run("verify", memory=memory)

    assert result.status == "success"
    assert tools.calls == [accepted]
    mismatch_payload = next(
        message.content or ""
        for message in memory.messages
        if message.tool_call_id == "write-mismatch"
    )
    assert "Decision declared run_command" in mismatch_payload
    assert all(event.tool_call_id != "write-mismatch" for event in memory.observation_events)


def test_legacy_protocol_observations_do_not_leave_session_reflection_locked() -> None:
    memory = MemoryState(session_id="reasoning-test", reflection_required=True)
    manager = ReasoningManager()
    manager.observe(
        ToolCall(id="read-1", name="read_file", arguments={"path": "app.py"}),
        ToolResult(
            ok=True,
            output="content",
            metadata={"path": "app.py", "start_line": 1, "end_line": 1},
        ),
        memory=memory,
        step=1,
        output_ref=None,
    )
    manager.observe(
        ToolCall(id="legacy-command", name="run_command", arguments={"command": "pytest -q"}),
        ToolResult(
            ok=False,
            output="",
            error="A decision record is required.",
            error_code="decision_validation_failed",
        ),
        memory=memory,
        step=2,
        output_ref=None,
    )
    command = ToolCall(
        id="command-accepted",
        name="run_command",
        arguments={"command": "python -m pytest -q"},
    )
    tools = FakeToolRegistry([ToolResult(ok=True, output="passed", metadata={"exit_code": 0})])
    agent, memory = controller(
        [
            ModelResponse(tool_calls=[decision("decision-1", "run_command")]),
            ModelResponse(tool_calls=[command]),
            ModelResponse(content="Verification passed."),
        ],
        tools,
        memory=memory,
    )

    result = agent.run("continue analysis", memory=memory)

    assert result.status == "success"
    assert tools.calls == [command]
    assert not memory.reflection_required


def test_legacy_repair_preserves_reflection_after_real_tool_failure() -> None:
    memory = MemoryState(session_id="reasoning-test", reflection_required=True)
    manager = ReasoningManager()
    manager.observe(
        ToolCall(id="failed-command", name="run_command", arguments={"command": "pytest -q"}),
        ToolResult(ok=False, output="", error="tests failed", metadata={"exit_code": 1}),
        memory=memory,
        step=1,
        output_ref=None,
    )
    manager.observe(
        ToolCall(id="legacy-command", name="run_command", arguments={"command": "pytest -q"}),
        ToolResult(
            ok=False,
            output="",
            error="A decision record is required.",
            error_code="decision_validation_failed",
        ),
        memory=memory,
        step=2,
        output_ref=None,
    )

    repaired = AgentController._repair_legacy_protocol_reflection(memory)

    assert not repaired
    assert memory.reflection_required


def test_reasoning_manager_bounds_history_compacts_decisions_and_redacts_secrets() -> None:
    memory = MemoryState(session_id="reasoning-test")
    manager = ReasoningManager(
        ReasoningConfig(recent_event_limit=2),
        secrets=("opaque-value",),
    )
    for index in range(3):
        manager.record(
            {
                "phase": "decision",
                "current_goal": "avoid api_key=visible-value",
                "summary": f"decision {index} contains opaque-value",
                "next_tool": "read_file",
                "expected_result": "file is observed",
            },
            memory=memory,
            step=index,
        )

    assert len(memory.reasoning_events) == 2
    assert len(memory.summary.key_decisions) == 3
    serialized = memory.model_dump_json()
    assert "opaque-value" not in serialized
    assert "visible-value" not in serialized
    assert "[REDACTED]" in serialized


def test_reasoning_summary_over_configured_limit_is_rejected() -> None:
    manager = ReasoningManager(ReasoningConfig(max_summary_chars=100))

    with pytest.raises(ValueError, match="exceeds 100"):
        manager.record(
            {
                "phase": "plan",
                "current_goal": "inspect",
                "summary": "x" * 101,
            },
            memory=MemoryState(session_id="reasoning-test"),
            step=0,
        )


def test_reasoning_and_observations_survive_session_checkpoint(tmp_path: Path) -> None:
    memory = MemoryState(session_id="reasoning-test")
    manager = ReasoningManager()
    event = manager.record(
        {
            "phase": "plan",
            "current_goal": "inspect the project",
            "summary": "Read the relevant files before changing anything.",
        },
        memory=memory,
        step=0,
    )
    observation = manager.observe(
        ToolCall(id="read-1", name="read_file", arguments={"path": "app.py"}),
        ToolResult(
            ok=True,
            output="content",
            metadata={"path": "app.py", "start_line": 1, "end_line": 10},
        ),
        memory=memory,
        step=1,
        output_ref=None,
    )
    store = MemoryStore.create(tmp_path)
    memory.session_id = store.session_id
    event.session_id = store.session_id
    observation.session_id = store.session_id
    store.save_state(memory, status="paused")

    restored = store.load_state()

    assert restored.reasoning_events[-1].summary == event.summary
    assert restored.observation_events[-1].event_id == observation.event_id


def test_observation_error_is_specific_and_audit_path_stays_executor_only() -> None:
    memory = MemoryState(session_id="reasoning-test")
    manager = ReasoningManager()
    result = ToolResult(
        ok=False,
        output="",
        error="File does not exist: missing.py",
        error_code="tool_error",
    )
    observation = manager.observe(
        ToolCall(id="read-1", name="read_file", arguments={"path": "missing.py"}),
        result,
        memory=memory,
        step=1,
        output_ref="file_snapshots/private.txt",
    )

    visible_result = manager.result_with_evidence(result, observation)

    assert (
        observation.result_summary
        == "read_file failed (tool_error): File does not exist: missing.py"
    )
    assert visible_result.metadata == {"evidence_ref": observation.event_id}
    assert observation.output_ref == "file_snapshots/private.txt"


@pytest.mark.parametrize(
    ("call", "result", "expected"),
    [
        (
            ToolCall(id="read", name="read_file", arguments={"path": "src/app.py"}),
            ToolResult(
                ok=True,
                output="content",
                metadata={"path": "src/app.py", "start_line": 4, "end_line": 9},
            ),
            "Read src/app.py:4-9.",
        ),
        (
            ToolCall(id="search", name="search_text", arguments={"query": "needle"}),
            ToolResult(
                ok=True,
                output="matches",
                metadata={
                    "matches": 2,
                    "match_locations": ["src/app.py:7", "tests/test_app.py:12"],
                },
            ),
            "Found 2 matching lines at src/app.py:7, tests/test_app.py:12.",
        ),
        (
            ToolCall(id="replace", name="edit_file", arguments={"path": "src/app.py"}),
            ToolResult(
                ok=True,
                output="changed",
                metadata={
                    "path": "src/app.py",
                    "changed_ranges": [[7, 8], [12, 12]],
                },
            ),
            "Committed snapshot-anchored edits at src/app.py:7-8, src/app.py:12-12.",
        ),
    ],
)
def test_file_observations_show_paths_and_line_numbers(
    call: ToolCall,
    result: ToolResult,
    expected: str,
) -> None:
    memory = MemoryState(session_id="reasoning-test")
    observation = ReasoningManager().observe(
        call,
        result,
        memory=memory,
        step=1,
        output_ref=None,
    )

    assert observation.result_summary == expected


def test_repeated_reflection_only_turns_prompt_for_concrete_progress() -> None:
    reflections = [
        ToolCall(
            id=f"reflection-{index}",
            name="record_decision",
            arguments={
                "phase": "reflection",
                "current_goal": "diagnose",
                "summary": f"reflection {index}",
            },
        )
        for index in range(3)
    ]
    responses = [*(ModelResponse(tool_calls=[item]) for item in reflections)]
    responses.append(ModelResponse(content="Stop."))
    model = FakeModelClient(responses)
    memory = MemoryState(session_id="reasoning-test")
    agent = AgentController(
        model_client=model,
        tool_registry=cast(ToolRegistry, FakeToolRegistry([])),
    )

    agent.run("diagnose", memory=memory)

    final_request = model.requests[-1]["messages"]
    assert any(
        "Too many reasoning-only turns" in (message.get("content") or "")
        for message in final_request
    )
    assert memory.reasoning_events[-1].phase == ReasoningPhase.REFLECTION


def test_final_assessment_meta_call_is_closed_before_the_next_model_request() -> None:
    final_assessment = ToolCall(
        id="final-1",
        name="record_decision",
        arguments={
            "phase": "final_assessment",
            "current_goal": "inspect the project",
            "summary": "The requested inspection is complete.",
            "evidence_refs": ["obs-read"],
            "confidence": 0.9,
        },
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[final_assessment]),
            ModelResponse(content="Inspection complete."),
        ]
    )
    memory = MemoryState(session_id="assessment-protocol")
    agent = AgentController(
        model_client=model,
        tool_registry=cast(ToolRegistry, FakeToolRegistry([])),
    )

    result = agent.run("inspect", memory=memory)

    assert result.status == "success"
    validate_tool_call_protocol([message.as_chat_message() for message in memory.messages])
    assert any(
        message.role == "tool" and message.tool_call_id == "final-1" for message in memory.messages
    )


def test_failed_action_requires_reflection_before_another_mutation() -> None:
    first = ToolCall(id="command-1", name="run_command", arguments={"command": "pytest -q"})
    blocked_call = ToolCall(
        id="command-2",
        name="run_command",
        arguments={"command": "pytest -q"},
    )
    retry = ToolCall(
        id="command-3",
        name="run_command",
        arguments={"command": "pytest -q --maxfail=1"},
    )
    reflection = ToolCall(
        id="reflection-1",
        name="record_decision",
        arguments={
            "phase": "reflection",
            "current_goal": "diagnose the failed test",
            "summary": "The verification command failed, so inspect the failure before retrying.",
            "evidence_refs": ["obs-failed"],
        },
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=False, output="", error="failed", metadata={"exit_code": 1}),
            ToolResult(ok=True, output="passed", metadata={"exit_code": 0}),
        ]
    )
    memory = MemoryState(session_id="reasoning-test")
    agent = AgentController(
        model_client=FakeModelClient(
            [
                ModelResponse(tool_calls=[decision("decision-1", "run_command"), first]),
                ModelResponse(tool_calls=[decision("decision-2", "run_command"), blocked_call]),
                ModelResponse(tool_calls=[reflection]),
                ModelResponse(tool_calls=[decision("decision-3", "run_command"), retry]),
                ModelResponse(content="Verification recovered."),
            ]
        ),
        tool_registry=cast(ToolRegistry, tools),
    )

    result = agent.run("run verification", memory=memory)

    assert result.status == "success"
    assert tools.calls == [first, retry]
    assert all(event.tool_call_id != "command-2" for event in memory.observation_events)
    blocked_payload = next(
        message.content or "" for message in memory.messages if message.tool_call_id == "command-2"
    )
    assert "Record a reflection in a separate turn" in blocked_payload
    assert not memory.reflection_required


def test_task_decision_cannot_bypass_independent_read_only_approval(tmp_path: Path) -> None:
    memory = MemoryState(session_id="reasoning-approval")
    approval = ApprovalEngine(
        mode=ApprovalMode.READ_ONLY,
        normalizer=RequestNormalizer(tmp_path),
        risk_analyzer=RiskAnalyzer(),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=NonInteractiveHumanApprover(),
    )
    registry = create_default_registry(tmp_path, approval_engine=approval)
    write = ToolCall(
        id="write-1",
        name="write_file",
        arguments={"path": "app.py", "content": "blocked"},
    )
    agent = AgentController(
        model_client=FakeModelClient(
            [
                ModelResponse(tool_calls=[decision("decision-1", "write_file"), write]),
                ModelResponse(content="The write was denied."),
            ]
        ),
        tool_registry=registry,
    )

    agent.run("write app.py", memory=memory)

    assert not (tmp_path / "app.py").exists()
    assert memory.reasoning_events[-1].next_action
    assert memory.reasoning_events[-1].next_action.tool_name == "write_file"
    assert not memory.observation_events[-1].ok
    assert memory.reflection_required
