"""Turn-group-aware rule compaction for recent raw conversation memory."""

from dataclasses import dataclass

from repo_rivet.memory.models import MemoryState, Message


@dataclass(slots=True)
class TurnGroup:
    """An indivisible assistant Tool Call and all of its Tool Results, or one plain message."""

    messages: list[Message]


class ConversationCompactor:
    """Discard raw groups only after their deterministic facts have entered summaries."""

    def compact_if_needed(
        self,
        memory: MemoryState,
        *,
        pressure: str,
    ) -> int:
        over_message_limit = len(memory.messages) > max(14, memory.config.recent_message_limit)
        if pressure == "normal" and not over_message_limit:
            return 0
        return self.compact(memory, aggressive=pressure in {"aggressive", "overflow"})

    def compact(
        self,
        memory: MemoryState,
        *,
        aggressive: bool,
        recovery_level: int = 0,
    ) -> int:
        groups = group_messages(memory.messages)
        if not groups:
            return 0

        target_messages = memory.config.recent_message_limit
        if aggressive:
            target_messages = max(2, target_messages // 2)
        if recovery_level >= 2:
            target_messages = max(1, target_messages // 2)

        kept: list[TurnGroup] = []
        kept_count = 0
        for group in reversed(groups):
            group_size = len(group.messages)
            if kept and kept_count + group_size > target_messages:
                break
            kept.append(group)
            kept_count += group_size

        kept.reverse()
        new_messages = [message for group in kept for message in group.messages]
        removed = len(memory.messages) - len(new_messages)
        if removed <= 0 and recovery_level < 2:
            return 0

        memory.messages = new_messages
        if recovery_level >= 2:
            self._shrink_current_tool_results(memory)
        if removed > 0:
            memory.compaction_count += 1
        return removed

    @staticmethod
    def _shrink_current_tool_results(memory: MemoryState) -> None:
        limit = max(500, memory.config.max_tool_output_chars // 4)
        for message in memory.messages:
            if message.role == "tool" and message.content and len(message.content) > limit:
                marker = "\n... aggressively truncated after context overflow ...\n"
                side = max(1, (limit - len(marker)) // 2)
                message.content = f"{message.content[:side]}{marker}{message.content[-side:]}"


def group_messages(messages: list[Message]) -> list[TurnGroup]:
    """Group every assistant Tool Call with all immediately following Tool Results."""
    groups: list[TurnGroup] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        grouped = [message]
        index += 1
        if message.role == "assistant" and message.tool_calls:
            while index < len(messages) and messages[index].role == "tool":
                grouped.append(messages[index])
                index += 1
        groups.append(TurnGroup(grouped))
    return groups
