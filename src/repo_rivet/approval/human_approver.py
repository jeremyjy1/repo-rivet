"""Interactive and fail-closed non-interactive approval adapters."""

import json
import select
import sys
from collections.abc import Callable
from typing import Protocol

from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from repo_rivet.approval.models import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
    LLMReviewResult,
    NonInteractivePolicy,
)


class HumanApprover(Protocol):
    def ask(
        self,
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None = None,
    ) -> ApprovalDecision:
        """Obtain or synthesize a human approval decision."""
        ...


class TerminalHumanApprover:
    """Show a bounded request summary and accept an explicit terminal choice."""

    def __init__(
        self,
        console: Console,
        *,
        reader: Callable[[str], str] | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.console = console
        self.reader = reader
        self.timeout_seconds = timeout_seconds

    def ask(
        self,
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None = None,
    ) -> ApprovalDecision:
        self.console.print(self._build_panel(request, llm_review=llm_review))
        prompt = "Select approval option [1-5]"
        while True:
            try:
                choice = self._read_choice(prompt)
            except (EOFError, KeyboardInterrupt):
                choice = "5"
            if choice is None:
                return ApprovalDecision(
                    action=ApprovalAction.DENY,
                    source="human_timeout",
                    reason="approval timed out without a user decision",
                    risk_level=request.assessment.level,
                    request_fingerprint=request.fingerprint,
                )
            choice = choice.strip().lower()
            decision = self._decision_for_choice(request, choice)
            if decision is not None:
                return decision
            self.console.print("Enter a number from 1 to 5.", style="yellow")

    @staticmethod
    def _build_panel(
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None,
    ) -> Panel:
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Tool", request.tool_name)
        summary.add_row("Risk", request.assessment.level.name)
        summary.add_row("Request", request.fingerprint[:12])

        reasons = Text()
        for index, reason in enumerate(request.assessment.reasons):
            if index:
                reasons.append("\n")
            reasons.append(f"• {reason}")
        if not reasons:
            reasons.append("• No deterministic reason was provided.")

        arguments = Text(
            json.dumps(
                request.normalized_arguments,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        sections: list[object] = [
            summary,
            Text("\nRisk reasons", style="bold"),
            reasons,
            Text("\nNormalized request", style="bold"),
            arguments,
        ]
        if llm_review is not None:
            review = Text()
            review.append(
                f"{llm_review.decision.upper()} · confidence {llm_review.confidence:.2f}\n",
                style="bold",
            )
            review.append(llm_review.reason)
            sections.extend((Text("\nLLM review", style="bold"), review))

        options = Table.grid(padding=(0, 2))
        options.add_column(style="bold cyan", justify="right")
        options.add_column()
        options.add_row("1", "Approve once")
        options.add_row("2", "Approve this exact request for the session")
        options.add_row("3", "Deny once")
        options.add_row("4", "Deny this exact request for the session")
        options.add_row("5", "Abort agent")
        sections.extend((Text("\nOptions", style="bold"), options))
        return Panel(Group(*sections), title="Approval Required", border_style="yellow")

    def _read_choice(self, prompt: str) -> str | None:
        if self.reader is not None:
            return self.reader(prompt)
        if self.timeout_seconds is None or not sys.stdin.isatty():
            return Prompt.ask(prompt, console=self.console)
        self.console.print(f"{prompt}: ", end="")
        try:
            readable, _, _ = select.select([sys.stdin], [], [], self.timeout_seconds)
        except (OSError, ValueError):
            return Prompt.ask(prompt, console=self.console)
        if not readable:
            self.console.print()
            return None
        return sys.stdin.readline()

    @staticmethod
    def _decision_for_choice(
        request: ApprovalRequest,
        choice: str,
    ) -> ApprovalDecision | None:
        choices = {
            "1": (ApprovalAction.ALLOW, ApprovalScope.ONCE, False, "approved by user"),
            "2": (
                ApprovalAction.ALLOW,
                ApprovalScope.SESSION_EXACT,
                False,
                "approved exact request for this session",
            ),
            "3": (ApprovalAction.DENY, ApprovalScope.ONCE, False, "denied by user"),
            "4": (
                ApprovalAction.DENY,
                ApprovalScope.SESSION_EXACT,
                False,
                "denied exact request for this session",
            ),
            "5": (ApprovalAction.DENY, ApprovalScope.ONCE, True, "agent aborted by user"),
        }
        selected = choices.get(choice)
        if selected is None:
            return None
        action, scope, abort_agent, reason = selected
        return ApprovalDecision(
            action=action,
            source="human",
            reason=reason,
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
            scope=scope,
            abort_agent=abort_agent,
        )


class NonInteractiveHumanApprover:
    """Never infer approval when no human input channel exists."""

    def __init__(self, policy: NonInteractivePolicy = NonInteractivePolicy.DENY) -> None:
        self.policy = policy

    def ask(
        self,
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None = None,
    ) -> ApprovalDecision:
        if self.policy == NonInteractivePolicy.FAIL:
            reason = "human approval is required but unavailable in non-interactive mode"
        else:
            reason = "request denied because non-interactive mode cannot obtain approval"
        return ApprovalDecision(
            action=ApprovalAction.DENY,
            source="non_interactive_policy",
            reason=reason,
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
            abort_agent=self.policy == NonInteractivePolicy.FAIL,
        )
