"""Skill discovery, validation, and activation errors."""


class SkillError(ValueError):
    """Base error for locally managed skills."""


class SkillNotFoundError(SkillError):
    """Raised when a requested skill ID is unavailable."""


class SkillValidationError(SkillError):
    """Raised when a SKILL.md file is malformed or unsafe."""


class SkillStaleError(SkillError):
    """Raised when a pinned skill changed after activation."""
