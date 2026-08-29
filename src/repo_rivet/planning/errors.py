"""Typed planning-workflow boundary errors."""


class PlanModeViolation(ValueError):
    """A model requested a capability that does not exist in Plan Mode."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"{tool_name} is unavailable in Plan Mode. Planning may inspect the workspace "
            "and submit or update a plan, but cannot execute commands, verification, or "
            "file changes."
        )
        self.tool_name = tool_name
