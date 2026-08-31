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
    assert "Cache-epoch checkpoint" in messages[2]["content"]
    assert "Structured state at epoch start" in messages[2]["content"]
    assert f"src/app.py: {'a' * 64}" in messages[2]["content"]
    assert any("pytest still fails" in str(message.get("content")) for message in messages)
    assert any(message.get("content") == "recent decision" for message in messages)


def test_build_keeps_exact_prior_request_prefix_and_only_appends_new_turns() -> None:
    memory = make_memory()
    manager = ContextManager()
    first = manager.build(
        memory=memory,
        state_summary="first state",
        remaining_steps=29,
        tools=[],
    )
    checkpoint = memory.context_checkpoint
    memory.messages.append(Message(role="assistant", content="first response"))

    memory.start_task(
        task="also add unit tests",
        workspace="/workspace",
        system_prompt=SYSTEM_PROMPT,
        safety_rules=["stay inside workspace"],
        completion_rules=["verify after changes"],
        max_steps=30,
    )
    second = manager.build(
        memory=memory,
        state_summary="second state",
        remaining_steps=28,
        tools=[],
    )

    assert second[: len(first)] == first
    assert memory.context_checkpoint == checkpoint
    assert "also add unit tests" not in second[1]["content"]
    assert [message.get("content") for message in second[-2:]] == [
        "first response",
        "also add unit tests",
    ]
    assert second[: len(first) + 1] == [
        *first,
        {"role": "assistant", "content": "first response"},
    ]
    assert "first state" in second[2]["content"]
    assert "second state" not in str(second)


def test_tool_turn_is_appended_after_exact_prior_request_without_reordering() -> None:
    memory = make_memory()
    manager = ContextManager()
    first = manager.build(
        memory=memory,
        state_summary="first state",
        remaining_steps=29,
        tools=[],
    )
    assistant = Message(
        role="assistant",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"app.py"}'},
            }
        ],
    )
    tool_result = Message(role="tool", tool_call_id="call-1", content="observed content")
    memory.messages.extend([assistant, tool_result])

    second = manager.build(
        memory=memory,
        state_summary="second state",
        remaining_steps=28,
        tools=[],
    )

    assert second[: len(first)] == first
    assert second[len(first) :] == [
        assistant.as_chat_message(),
        tool_result.as_chat_message(),
    ]


def test_normal_pressure_does_not_compact_append_only_history() -> None:
    memory = make_memory(config=MemoryConfig(recent_message_limit=4))
    for index in range(20):
        memory.messages.append(Message(role="assistant", content=f"short message {index}"))

    messages = ContextManager().build(
        memory=memory,
        state_summary="state",
        remaining_steps=5,
        tools=[],
    )

    assert len(memory.messages) == 21
    assert memory.compaction_count == 0
    assert any(message.get("content") == "short message 0" for message in messages)


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
    recent_trace = structured.partition("Recent auditable trace:")[2]
    assert "bounded decision 5" in recent_trace
    assert "bounded decision 0" not in recent_trace


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
    memory = make_memory(
        config=MemoryConfig(
            recent_message_limit=4,
            max_context_tokens=4_000,
            reserved_output_tokens=1_000,
            reserved_tool_result_tokens=100,
            safety_margin_ratio=0.05,
        )
    )
    memory.summary.unresolved_issues.append("failing test must remain")
    for index in range(20):
        memory.messages.append(
            Message(role="assistant", content=f"old message {index} " + "x" * 800)
        )

    manager = ContextManager()
    messages = manager.build(
        memory=memory,
        state_summary="state",
        remaining_steps=5,
        tools=[],
    )

    assert len(memory.messages) <= 4
    assert memory.compaction_count == 1
    assert manager.last_pressure != "normal"
    assert "preserve this original task exactly" in messages[1]["content"]
    assert any("failing test must remain" in str(message.get("content")) for message in messages)


def test_compaction_does_not_split_assistant_tool_pair() -> None:
    memory = make_memory(
        config=MemoryConfig(
            recent_message_limit=4,
            max_context_tokens=4_000,
            reserved_output_tokens=1_000,
            reserved_tool_result_tokens=100,
            safety_margin_ratio=0.05,
        )
    )
    for index in range(12):
        memory.messages.append(Message(role="assistant", content=f"old {index} " + "x" * 800))
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

    with pytest.raises(ValueError, match="exceed the active prompt budget"):
        ContextManager().build(
            memory=memory,
            state_summary="state",
            remaining_steps=10,
            tools=[],
        )


def test_large_model_window_still_compacts_at_active_prompt_cost_limit() -> None:
    memory = make_memory(
        config=MemoryConfig(
            recent_message_limit=4,
            max_context_tokens=1_048_576,
            active_prompt_limit=8_000,
        )
    )
    for index in range(30):
        memory.messages.append(
            Message(role="assistant", content=f"costly history {index} " + "x" * 1_000)
        )

    manager = ContextManager()
    messages = manager.build(
        memory=memory,
        state_summary="latest state after cost compaction",
        remaining_steps=5,
        tools=[],
    )

    assert memory.compaction_count == 1
    assert len(memory.messages) <= 4
    assert memory.context_checkpoint is not None
    assert "latest state after cost compaction" in memory.context_checkpoint
    assert manager.last_request_tokens <= 8_000
    assert messages[2]["content"] == memory.context_checkpoint


def test_cache_checkpoint_bounds_long_structured_summary_entries() -> None:
    memory = make_memory(
        config=MemoryConfig(
            max_context_tokens=1_048_576,
            active_prompt_limit=16_000,
        )
    )
    memory.summary.key_decisions = [f"decision {index}: " + "x" * 1_000 for index in range(100)]

    manager = ContextManager()
    manager.build(
        memory=memory,
        state_summary="state",
        remaining_steps=5,
        tools=[],
    )

    assert memory.context_checkpoint is not None
    assert "80 earlier entries" in memory.context_checkpoint
    assert "decision 99" in memory.context_checkpoint
    assert "decision 0:" not in memory.context_checkpoint
    assert manager.last_request_tokens <= 16_000
