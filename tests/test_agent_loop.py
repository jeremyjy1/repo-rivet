from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from repo_rivet.agent.controller import AgentController
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy
from repo_rivet.llm.base import ModelContextLengthError, ModelResponse
from repo_rivet.llm.openai_compatible import ModelRequestError
from repo_rivet.llm.parser import ResponseParseError
from repo_rivet.llm.protocol import validate_tool_call_protocol
from repo_rivet.memory.models import MemoryState, Message
from repo_rivet.memory.store import MemoryStore
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
    model = FakeModelClient([ModelResponse(tool_calls=[read]), ModelResponse(content="Done.")])
    tools = FakeToolRegistry([ToolResult(ok=True, output="content")])

    result = controller(model, tools).run("inspect the project")

    assert result.status == "success"
    assert result.summary == "Done."
    assert result.step_count == 2
    assert result.tool_call_count == 1


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


def test_resume_removes_legacy_empty_assistant_before_request() -> None:
    model = FakeModelClient([ModelResponse(content="The resumed task can continue.")])
    tools = FakeToolRegistry([])
    events = RecordingSink()
    memory = MemoryState(
        session_id="legacy-empty-history",
        messages=[
            Message(role="user", content="previous task"),
            Message(role="assistant", content=None),
        ],
    )

    result = controller(model, tools, event_logger=events).run(
        "continue after resume",
        memory=memory,
    )

    assert result.status == "success"
    validate_tool_call_protocol(model.requests[0]["messages"])
    assert not any(
        message.role == "assistant" and not message.is_valid_provider_message()
        for message in memory.messages
    )
    repair_events = [data for event, data in events.events if event == "invalid_history_repaired"]
    assert repair_events == [{"removed_empty_assistant_messages": 1}]


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
            ModelResponse(tool_calls=[decision("d2", "run_verification"), compile_check]),
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

    result = controller(
        model,
        tools,
        termination=TerminationPolicy(TerminationConfig(max_steps=3)),
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


def test_verification_pass_on_final_model_step_finishes_instead_of_stopping() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    run_check = call("verify-tests", "run_verification", {"check_id": "tests"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[verification_plan(), decision("d1", "write_file"), write]),
            ModelResponse(tool_calls=[decision("d2", "run_verification"), run_check]),
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
            ModelResponse(tool_calls=[decision("d2", "run_verification"), first_check]),
            ModelResponse(tool_calls=[repeated_check]),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=True, output="written"),
            passed_verification_result(),
        ]
    )

    result = controller(model, tools).run("implement and verify")

    assert result.status == "success"
    assert result.verification_status.value == "passed"
    assert [item.name for item in tools.calls] == ["write_file", "run_verification"]
    assert model.requests[2]["tools"] == []


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
            ModelResponse(tool_calls=[decision("d2", "run_verification"), run_check]),
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


def test_controller_persists_verifying_and_failed_scheduled_check_before_abort() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    memory = MemoryState(session_id="verification-abort")

    class StatusInspectingRegistry(FakeToolRegistry):
        status_at_verification: str | None = None

        def execute(self, tool_call: ToolCall) -> ToolResult:
            if tool_call.name == "run_verification":
                self.status_at_verification = memory.status
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

    result = controller(model, tools).run("fix", memory=memory)

    assert result.status == "stopped"
    assert tools.status_at_verification == "verifying"
    assert memory.verification_results["tests"].status.value == "error"
    assert memory.status == "stopped"


def test_missing_verification_plan_gets_one_recovery_then_finishes_incomplete() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[decision("d1", "write_file"), write]),
            ModelResponse(content="Done without a plan."),
            ModelResponse(content="Still done without a plan."),
        ]
    )
    tools = FakeToolRegistry([ToolResult(ok=True, output="written")])
    memory = MemoryState(session_id="missing-plan")

    result = controller(model, tools).run("fix the bug", memory=memory)

    assert result.status == "incomplete"
    assert result.reason == "no executable verification plan was registered after one recovery"
    assert len(model.requests) == 3
    assert memory.verification_plan_recovery_attempts == 1
    assert memory.candidate_final_assessment
    recovery = model.requests[2]["messages"]
    assert any("verification_plan_missing" in str(message.get("content")) for message in recovery)


def test_provider_block_preserves_candidate_assessment_and_error_details() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
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
            ModelResponse(tool_calls=[decision("d1", "write_file"), write]),
            ModelResponse(content="Candidate final answer."),
            provider_error,
        ]
    )
    memory = MemoryState(session_id="blocked-provider")

    result = controller(
        model,
        FakeToolRegistry([ToolResult(ok=True, output="written")]),
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


def test_missing_durable_reasoning_for_tool_history_disables_thinking_on_resume() -> None:
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
    assert model.requests[0]["options"].thinking_enabled is False


def test_agent_stops_repeated_identical_tool_calls() -> None:
    repeated = call("1", "read_file", {"path": "app.py"})
    model = FakeModelClient([ModelResponse(tool_calls=[repeated])] * 3)
    tools = FakeToolRegistry([ToolResult(ok=True, output="content")] * 3)
    termination = TerminationPolicy(TerminationConfig(max_repeated_tool_calls=3))

    result = controller(model, tools, termination=termination).run("inspect")

    assert result.status == "stopped"
    assert "repeated identical tool call" in (result.reason or "")
    assert result.tool_call_count == 3


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
    assert any("Durable subsequent user requirements" in str(content) for content in contents)
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
