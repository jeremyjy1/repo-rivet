from typing import cast

from repo_rivet.agent.controller import AgentController
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy
from repo_rivet.llm.base import ModelResponse
from repo_rivet.llm.parser import ResponseParseError
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.tools.registry import ToolRegistry
from tests.fakes import FakeModelClient, FakeToolRegistry


def call(call_id: str, name: str, arguments: dict) -> ToolCall:  # type: ignore[type-arg]
    return ToolCall(id=call_id, name=name, arguments=arguments)


def controller(
    model: FakeModelClient,
    tools: FakeToolRegistry,
    *,
    termination: TerminationPolicy | None = None,
) -> AgentController:
    return AgentController(
        model_client=model,
        tool_registry=cast(ToolRegistry, tools),
        termination_policy=termination,
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
            ModelResponse(tool_calls=[write]),
            ModelResponse(content="Finished too early."),
            ModelResponse(tool_calls=[verify]),
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
