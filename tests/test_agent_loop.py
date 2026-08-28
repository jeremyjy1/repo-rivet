from typing import cast

from repo_rivet.agent.controller import AgentController
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy
from repo_rivet.llm.base import ModelContextLengthError, ModelResponse
from repo_rivet.llm.parser import ResponseParseError
from repo_rivet.memory.models import MemoryState, Message
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.tools.registry import ToolRegistry
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


def test_agent_refuses_early_finish_until_change_is_verified() -> None:
    write = call("1", "write_file", {"path": "app.py", "content": "new"})
    verify = call("2", "run_command", {"command": "pytest -q"})
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[decision("d1", "write_file"), write]),
            ModelResponse(content="Finished too early."),
            ModelResponse(tool_calls=[decision("d2", "run_command"), verify]),
            ModelResponse(content="Implemented and tested."),
        ]
    )
    tools = FakeToolRegistry(
        [
            ToolResult(ok=True, output="written"),
            ToolResult(ok=True, output="passed", metadata={"exit_code": 0}),
        ]
    )

    result = controller(model, tools).run("fix the bug")

    assert result.status == "success"
    assert result.summary == "Implemented and tested."
    assert result.modified_files == ("app.py",)
    assert result.verification_success
    third_request = model.requests[2]["messages"]
    assert any("Files changed" in (message.get("content") or "") for message in third_request)


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
    assert result.reason == "model API failed after retries: RuntimeError"


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
    assert "continue" in task_spec
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
