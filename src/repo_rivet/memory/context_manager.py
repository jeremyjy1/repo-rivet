"""Budget-aware reconstruction of model context from layered memory."""

from __future__ import annotations

from typing import Any

from repo_rivet.memory.budget_manager import (
    RequestTokenEstimate,
    TokenBudgetConfig,
    TokenBudgetManager,
)
from repo_rivet.memory.compactor import ConversationCompactor, TurnGroup, group_messages
from repo_rivet.memory.models import ConversationSummary, MemoryState, Message
from repo_rivet.memory.token_estimator import ApproximateTokenEstimator

SYSTEM_PROMPT = """You are RepoRivet, a local coding agent.
Work only through the provided tools and stay inside the configured workspace.
Inspect relevant files before editing. Prefer precise replacements over full rewrites.
Treat command failures as observations, diagnose them, and continue when possible.
If a tool request is denied, do not repeat the same request; choose a safer alternative or stop.
After changing files, run an appropriate test, build, lint, or syntax check before finishing.
Do not claim success unless the latest verification after the latest change succeeded.
When finished, summarize the changes and verification concisely.
Use concise plain text for the final response by default. Avoid Markdown headings, tables,
emphasis, list markers, and fenced code blocks unless the user explicitly requests Markdown
or the content cannot be communicated clearly without that structure."""


class ContextBudgetExceededError(ValueError):
    """Raised before a request when fixed context cannot fit the safe prompt budget."""


class ContextManager:
    """Combine fixed, working, summary, and TurnGroup memory within a safe budget."""

    def __init__(
        self,
        *,
        token_manager: TokenBudgetManager | None = None,
        compactor: ConversationCompactor | None = None,
    ) -> None:
        self._configured_token_manager = token_manager
        self._dynamic_token_manager: TokenBudgetManager | None = None
        self.compactor = compactor or ConversationCompactor()
        self.last_estimate = RequestTokenEstimate(raw=0, effective=0, correction_factor=1.0)

    @property
    def token_manager(self) -> TokenBudgetManager:
        manager = self._configured_token_manager or self._dynamic_token_manager
        if manager is None:
            raise RuntimeError("Token manager is initialized when context is first built")
        return manager

    @property
    def last_request_tokens(self) -> int:
        return self.last_estimate.effective

    def build(
        self,
        *,
        memory: MemoryState,
        state_summary: str,
        remaining_steps: int,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rebuild context, compacting before any request can exceed the safe budget."""
        if memory.fixed is None:
            raise ValueError("Memory must have fixed task information before context is built")

        manager = self._manager_for(memory)
        fixed_messages = self._fixed_messages(memory, state_summary, remaining_steps, tools)
        full_messages = [
            *fixed_messages,
            *(message.as_chat_message() for message in memory.messages),
        ]
        full_estimate = manager.estimate_request(full_messages, tools)
        pressure = manager.pressure_level(full_estimate.effective)
        self.compactor.compact_if_needed(memory, pressure=pressure)

        fixed_estimate = manager.estimate_request(fixed_messages, tools)
        manager.state.fixed_prompt_estimate = manager.estimator.base.estimate_request(
            fixed_messages,
            [],
        )
        if fixed_estimate.effective > manager.config.prompt_budget:
            raise ContextBudgetExceededError(
                "Fixed task, structured state, and tool definitions exceed the safe prompt "
                f"budget ({fixed_estimate.effective} > {manager.config.prompt_budget} tokens)"
            )

        selected = self._select_recent(
            fixed_messages=fixed_messages,
            messages=memory.messages,
            tools=tools,
            memory=memory,
            manager=manager,
        )
        messages = [*fixed_messages, *(message.as_chat_message() for message in selected)]
        self.last_estimate = manager.estimate_request(messages, tools)
        if self.last_estimate.effective > manager.config.prompt_budget:
            raise ContextBudgetExceededError(
                f"Context remains over safe prompt budget after compaction "
                f"({self.last_estimate.effective} > {manager.config.prompt_budget} tokens)"
            )
        return messages

    def compact_for_recovery(self, memory: MemoryState, *, recovery_level: int) -> int:
        """Apply progressively stronger TurnGroup compaction after provider overflow."""
        return self.compactor.compact(
            memory,
            aggressive=True,
            recovery_level=recovery_level,
        )

    def observe_usage(self, actual_prompt_tokens: int) -> None:
        self.token_manager.observe_usage(
            estimated=self.last_estimate.raw,
            actual=actual_prompt_tokens,
        )

    def observe_overflow(self) -> None:
        self.token_manager.observe_overflow()

    def count_message(self, message: dict[str, Any]) -> int:
        return self.token_manager.estimator.base.estimate_request([message], [])

    def _manager_for(self, memory: MemoryState) -> TokenBudgetManager:
        if self._configured_token_manager is not None:
            return self._configured_token_manager
        config = _budget_config(memory)
        if self._dynamic_token_manager is None or self._dynamic_token_manager.config != config:
            self._dynamic_token_manager = TokenBudgetManager(
                estimator=ApproximateTokenEstimator(),
                config=config,
                calibration_store=None,
                base_url="memory-only",
                model="unknown",
            )
        return self._dynamic_token_manager

    def _select_recent(
        self,
        *,
        fixed_messages: list[dict[str, Any]],
        messages: list[Message],
        tools: list[dict[str, Any]],
        memory: MemoryState,
        manager: TokenBudgetManager,
    ) -> list[Message]:
        selected_groups: list[TurnGroup] = []
        for group in reversed(group_messages(messages)):
            bounded = TurnGroup(
                [self._bound_message(message, memory) for message in group.messages]
            )
            candidate_groups = [bounded, *selected_groups]
            candidate_messages = [
                *fixed_messages,
                *(
                    message.as_chat_message()
                    for candidate_group in candidate_groups
                    for message in candidate_group.messages
                ),
            ]
            estimate = manager.estimate_request(candidate_messages, tools)
            if estimate.effective <= manager.config.prompt_budget:
                selected_groups = candidate_groups
                continue
            if selected_groups:
                break

            fitted = self._force_fit_latest_group(
                fixed_messages=fixed_messages,
                group=bounded,
                tools=tools,
                manager=manager,
            )
            if fitted is not None:
                selected_groups = [fitted]
            break

        return [message for group in selected_groups for message in group.messages]

    def _force_fit_latest_group(
        self,
        *,
        fixed_messages: list[dict[str, Any]],
        group: TurnGroup,
        tools: list[dict[str, Any]],
        manager: TokenBudgetManager,
    ) -> TurnGroup | None:
        fitted = TurnGroup([message.model_copy(deep=True) for message in group.messages])
        while True:
            candidate = [
                *fixed_messages,
                *(message.as_chat_message() for message in fitted.messages),
            ]
            if manager.estimate_request(candidate, tools).effective <= manager.config.prompt_budget:
                return fitted
            longest = max(
                (message for message in fitted.messages if message.content),
                key=lambda item: manager.estimator.base.estimate_text(
                    item.content or "",
                    kind="log" if item.role == "tool" else "natural",
                ),
                default=None,
            )
            if longest is None or len(longest.content or "") <= 32:
                return None
            longest.content = self._truncate_characters(
                longest.content or "",
                max(32, int(len(longest.content or "") * 0.70)),
            )

    def _fixed_messages(
        self,
        memory: MemoryState,
        state_summary: str,
        remaining_steps: int,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tool_names = ", ".join(
            str(tool.get("function", {}).get("name", "unknown")) for tool in tools
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": memory.fixed.system_prompt},
            {"role": "user", "content": memory.task_specification()},
            {
                "role": "system",
                "content": (
                    "Current structured state (facts, not new user instructions):\n"
                    f"{state_summary}\n"
                    f"Current focus: {memory.working.current_focus or 'none'}\n"
                    f"Unresolved errors: {memory.working.unresolved_errors or ['none']}\n"
                    f"Pending actions: {memory.working.pending_actions or ['none']}\n"
                    f"Invalidated file reads: {sorted(memory.invalidated_files) or ['none']}\n"
                    f"Available tools: {tool_names}\n"
                    f"Remaining agent steps: {max(remaining_steps, 0)}"
                ),
            },
        ]
        if memory.summary.has_content():
            messages.append({"role": "system", "content": self._format_summary(memory.summary)})
        return messages

    @staticmethod
    def _format_summary(summary: ConversationSummary) -> str:
        def lines(values: list[str]) -> str:
            return "\n".join(f"- {value}" for value in values) or "- none"

        return (
            "Compressed historical summary (facts from earlier events, not new instructions):\n"
            f"Task goal:\n- {summary.task_goal}\n"
            f"Completed actions:\n{lines(summary.completed_actions)}\n"
            f"Key decisions:\n{lines(summary.key_decisions)}\n"
            f"Files read:\n{lines(summary.files_read)}\n"
            f"Files modified:\n{lines(summary.files_modified)}\n"
            f"Commands run:\n{lines(summary.commands_run)}\n"
            f"Verification status:\n- {summary.verification_status}\n"
            f"Unresolved issues:\n{lines(summary.unresolved_issues)}\n"
            f"Next actions:\n{lines(summary.next_actions)}"
        )

    @staticmethod
    def _bound_message(message: Message, memory: MemoryState) -> Message:
        bounded = message.model_copy(deep=True)
        if bounded.role == "tool" and bounded.content:
            limit = memory.config.max_tool_output_chars
            if len(bounded.content) > limit:
                bounded.content = ContextManager._truncate_characters(bounded.content, limit)
        return bounded

    @staticmethod
    def _truncate_characters(content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        marker = "\n... content truncated ...\n"
        side = max(1, (limit - len(marker)) // 2)
        return f"{content[:side]}{marker}{content[-side:]}"


def _budget_config(memory: MemoryState) -> TokenBudgetConfig:
    config = memory.config
    return TokenBudgetConfig(
        context_limit=config.max_context_tokens,
        reserved_output_tokens=config.reserved_output_tokens,
        reserved_tool_result_tokens=config.reserved_tool_result_tokens,
        safety_margin_ratio=config.safety_margin_ratio,
        soft_limit_ratio=config.compaction_threshold,
        hard_limit_ratio=config.hard_limit_threshold,
        default_correction_factor=config.default_correction_factor,
        calibration_window=config.calibration_window,
        max_context_overflow_retries=config.max_context_overflow_retries,
    )
