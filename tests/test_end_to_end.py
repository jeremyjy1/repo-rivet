import shlex
import sys
from pathlib import Path

from repo_rivet.agent.controller import AgentController
from repo_rivet.llm.base import ModelResponse
from repo_rivet.storage.event_logger import EventLogger
from repo_rivet.tools.base import ToolCall
from repo_rivet.tools.registry import create_default_registry
from tests.fakes import FakeModelClient


def test_real_tools_complete_three_step_edit_and_verification_loop(tmp_path: Path) -> None:
    source_path = tmp_path / "discount.py"
    source_path.write_text(
        "def calculate_discount(price, rate):\n    return price * (1 - rate)\n",
        encoding="utf-8",
    )
    read = ToolCall(id="1", name="read_file", arguments={"path": "discount.py"})
    replace = ToolCall(
        id="2",
        name="replace_text",
        arguments={
            "path": "discount.py",
            "old_text": "    return price * (1 - rate)",
            "new_text": (
                '    if price < 0:\n        raise ValueError("price must not be negative")\n'
                "    return price * (1 - rate)"
            ),
            "expected_count": 1,
        },
    )
    verify = ToolCall(
        id="3",
        name="run_command",
        arguments={
            "command": f"{shlex.quote(sys.executable)} -m py_compile discount.py",
        },
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[read]),
            ModelResponse(tool_calls=[replace]),
            ModelResponse(tool_calls=[verify]),
            ModelResponse(content="Rejected negative prices and verified the module."),
        ]
    )
    log_path = tmp_path / "session.jsonl"
    controller = AgentController(
        model_client=model,
        tool_registry=create_default_registry(tmp_path),
        event_logger=EventLogger(log_path),
    )

    result = controller.run("Reject negative prices and verify the code")

    assert result.status == "success"
    assert result.tool_call_count == 3
    assert result.verification_success
    assert "raise ValueError" in source_path.read_text(encoding="utf-8")
    log_content = log_path.read_text(encoding="utf-8")
    assert '"event": "session_start"' in log_content
    assert '"event": "session_end"' in log_content
