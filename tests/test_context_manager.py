from copy import deepcopy

from repo_rivet.context.manager import SYSTEM_PROMPT, ContextManager


def test_build_includes_fixed_task_state_and_recent_history() -> None:
    manager = ContextManager(recent_message_limit=2)
    history = [
        {"role": "assistant", "content": "old"},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
        {"role": "assistant", "content": "recent"},
    ]

    messages = manager.build(
        task="fix bug",
        history=history,
        state_summary="No files changed.",
        remaining_steps=29,
    )

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "fix bug"}
    assert "Earlier context summary" in messages[2]["content"]
    assert messages[-1]["content"].endswith("Remaining agent steps: 29")
    assert any(message.get("content") == "recent" for message in messages)


def test_build_truncates_tool_output_without_mutating_history() -> None:
    manager = ContextManager(max_tool_content_chars=80)
    history = [{"role": "tool", "tool_call_id": "1", "content": "x" * 500}]
    original = deepcopy(history)

    messages = manager.build(
        task="task",
        history=history,
        state_summary="state",
        remaining_steps=1,
    )

    tool_message = next(message for message in messages if message["role"] == "tool")
    assert "content truncated" in tool_message["content"]
    assert len(tool_message["content"]) <= 80
    assert history == original
