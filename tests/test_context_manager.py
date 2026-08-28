import pytest

from repo_rivet.memory.context_manager import SYSTEM_PROMPT, ContextManager
from repo_rivet.memory.models import MemoryConfig, MemoryState, Message
from repo_rivet.reasoning.manager import ReasoningManager


def make_memory(*, config: MemoryConfig | None = None) -> MemoryState:
    memory = MemoryState(session_id="test-session", config=config or MemoryConfig())
    memory.start_task(
        task="preserve this original task exactly",
        workspace="/workspace",
        system_prompt=SYSTEM_PROMPT,
        safety_rules=["stay inside workspace"],
        completion_rules=["verify after changes"],
        max_steps=30,
    )
    return memory


def test_system_prompt_defaults_final_response_to_plain_text() -> None:
    assert "plain text for the final response by default" in SYSTEM_PROMPT
    assert "unless the user explicitly requests Markdown" in SYSTEM_PROMPT


def test_build_keeps_fixed_task_state_summary_and_recent_messages() -> None:
    memory = make_memory()
    memory.summary.files_modified.append("src/app.py")
    memory.summary.unresolved_issues.append("pytest still fails")
    memory.current_snapshots["src/app.py"] = "a" * 64
    memory.messages.append(Message(role="assistant", content="recent decision"))

    messages = ContextManager().build(
        memory=memory,
        state_summary="Verification required.",
        remaining_steps=29,
        tools=[],
    )

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "preserve this original task exactly" in messages[1]["content"]
    assert "Current structured state" in messages[2]["content"]
    assert f"src/app.py: {'a' * 64}" in messages[2]["content"]
    assert any("pytest still fails" in str(message.get("content")) for message in messages)
    assert any(message.get("content") == "recent decision" for message in messages)


def test_build_includes_only_bounded_recent_auditable_trace() -> None:
    memory = make_memory()
    manager = ReasoningManager()
    for index in range(6):
        manager.record(
            {
                "phase": "decision",
                "current_goal": "inspect",
                "summary": f"bounded decision {index}",
                "next_tool": "read_file",
                "expected_result": "file is observed",
            },
            memory=memory,
            step=index,
        )

    messages = ContextManager().build(
        memory=memory,
        state_summary="state",
        remaining_steps=10,
        tools=[],
    )

    structured = next(
        str(message.get("content"))
        for message in messages
        if "Recent auditable trace" in str(message.get("content"))
    )
    assert "bounded decision 5" in structured
    assert "bounded decision 0" not in structured


def test_build_truncates_large_tool_output_without_mutating_memory() -> None:
    memory = make_memory(
        config=MemoryConfig(
            max_tool_output_chars=1_000,
            max_context_tokens=4_000,
            reserved_output_tokens=1_000,
            reserved_tool_result_tokens=100,
            safety_margin_ratio=0.05,
        )
    )
    original = "head\n" + "x" * 3_000 + "\nimportant tail"
    memory.messages.extend(
        [
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "run_command", "arguments": "{}"},
                    }
                ],
            ),
            Message(role="tool", tool_call_id="call-1", content=original),
        ]
    )

    manager = ContextManager()
    messages = manager.build(
        memory=memory,
        state_summary="state",
        remaining_steps=10,
        tools=[],
    )

    tool_message = next(message for message in messages if message["role"] == "tool")
    assert "content truncated" in tool_message["content"]
    assert "important tail" in tool_message["content"]
    assert memory.messages[-1].content == original
    assert manager.last_request_tokens <= manager.token_manager.config.prompt_budget


def test_compaction_preserves_original_task_and_structured_unresolved_issue() -> None:
    memory = make_memory(config=MemoryConfig(recent_message_limit=4))
    memory.summary.unresolved_issues.append("failing test must remain")
    for index in range(20):
        memory.messages.append(Message(role="assistant", content=f"old message {index}"))

    messages = ContextManager().build(
        memory=memory,
        state_summary="state",
        remaining_steps=5,
        tools=[],
    )

    assert len(memory.messages) <= 4
    assert memory.compaction_count == 1
    assert "preserve this original task exactly" in messages[1]["content"]
    assert any("failing test must remain" in str(message.get("content")) for message in messages)


def test_compaction_does_not_split_assistant_tool_pair() -> None:
    memory = make_memory(config=MemoryConfig(recent_message_limit=4))
    for index in range(12):
        memory.messages.append(Message(role="assistant", content=f"old {index}"))
    memory.messages.extend(
        [
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    },
                ],
            ),
            Message(role="tool", tool_call_id="call-1", content="result one"),
            Message(role="tool", tool_call_id="call-2", content="result two"),
        ]
    )

    messages = ContextManager().build(
        memory=memory,
        state_summary="state",
        remaining_steps=5,
        tools=[],
    )

    assistant_index = next(index for index, item in enumerate(messages) if item.get("tool_calls"))
    tool_indexes = [index for index, item in enumerate(messages) if item["role"] == "tool"]
    assert tool_indexes == [assistant_index + 1, assistant_index + 2]
    assert [messages[index]["tool_call_id"] for index in tool_indexes] == ["call-1", "call-2"]


def test_fixed_context_larger_than_configured_window_is_rejected() -> None:
    memory = make_memory(
        config=MemoryConfig(
            max_context_tokens=1_000,
            reserved_output_tokens=100,
            reserved_tool_result_tokens=100,
            safety_margin_ratio=0.05,
        )
    )
    assert memory.fixed is not None
    memory.fixed.original_task = "large fixed task " * 2_000

    with pytest.raises(ValueError, match="exceed the safe prompt budget"):
        ContextManager().build(
            memory=memory,
            state_summary="state",
            remaining_steps=10,
            tools=[],
        )
