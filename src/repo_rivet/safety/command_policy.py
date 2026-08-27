"""Conservative validation for commands executed without a shell."""

import shlex
from pathlib import Path


class CommandPolicyError(ValueError):
    """Raised when a command is empty, unsupported, or clearly destructive."""


class CommandPolicy:
    """Parse commands into argument lists and reject high-risk invocations."""

    _SHELL_OPERATORS = frozenset({";", "&&", "||", "|", "&", ">", ">>", "<", "<<"})
    _BLOCKED_EXECUTABLES = frozenset(
        {
            "dd",
            "fdisk",
            "halt",
            "parted",
            "poweroff",
            "reboot",
            "rm",
            "shutdown",
            "sudo",
            "su",
        }
    )
    _SHELL_EXECUTABLES = frozenset({"bash", "cmd", "fish", "powershell", "pwsh", "sh", "zsh"})

    def __init__(self, max_command_length: int = 4096) -> None:
        if max_command_length <= 0:
            raise ValueError("max_command_length must be positive")
        self._max_command_length = max_command_length

    def validate(self, command: str) -> tuple[str, ...]:
        """Return a shell-free argv tuple when the command passes policy checks."""
        arguments = self.parse(command)
        executable = Path(arguments[0]).name.lower()
        if executable in self._BLOCKED_EXECUTABLES or executable.startswith("mkfs"):
            raise CommandPolicyError(f"Blocked executable: {executable}")
        if executable == "git":
            self._validate_git(list(arguments))
        return arguments

    def parse(self, command: str) -> tuple[str, ...]:
        """Reject shell syntax and return canonical argv before approval analysis."""
        if not command.strip():
            raise CommandPolicyError("Command must not be empty")
        if len(command) > self._max_command_length:
            raise CommandPolicyError("Command exceeds the maximum allowed length")
        if "\n" in command or "\r" in command:
            raise CommandPolicyError("Multiline commands are not supported")

        try:
            arguments = shlex.split(command, posix=True)
        except ValueError as error:
            raise CommandPolicyError(f"Could not parse command: {error}") from None

        if not arguments:
            raise CommandPolicyError("Command must not be empty")
        if any(argument in self._SHELL_OPERATORS for argument in arguments):
            raise CommandPolicyError("Shell operators are not supported")

        executable = Path(arguments[0]).name.lower()
        if executable in self._SHELL_EXECUTABLES and self._uses_command_string(
            executable, arguments
        ):
            raise CommandPolicyError(f"Shell command strings are not allowed: {executable}")
        return tuple(arguments)

    @staticmethod
    def _uses_command_string(executable: str, arguments: list[str]) -> bool:
        if executable == "cmd":
            return any(argument.lower() in {"/c", "/k"} for argument in arguments[1:])
        return "-c" in arguments[1:]

    @staticmethod
    def _validate_git(arguments: list[str]) -> None:
        normalized = {argument.lower() for argument in arguments[1:]}
        if "reset" in normalized and "--hard" in normalized:
            raise CommandPolicyError("git reset --hard is not allowed")
        if "clean" in normalized and any(
            argument.startswith("-") and "f" in argument.lower() for argument in arguments[1:]
        ):
            raise CommandPolicyError("git clean with force is not allowed")
