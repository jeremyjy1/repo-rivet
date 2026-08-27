import pytest

from repo_rivet.safety.command_policy import CommandPolicy, CommandPolicyError


@pytest.fixture
def policy() -> CommandPolicy:
    return CommandPolicy()


def test_accept_safe_command(policy: CommandPolicy) -> None:
    assert policy.validate('pytest -q "tests/unit tests"') == (
        "pytest",
        "-q",
        "tests/unit tests",
    )


@pytest.mark.parametrize("command", ["", "   ", "pytest\nrm file", "pytest && echo done"])
def test_reject_empty_multiline_or_shell_command(policy: CommandPolicy, command: str) -> None:
    with pytest.raises(CommandPolicyError):
        policy.validate(command)


@pytest.mark.parametrize(
    "command",
    [
        "rm file.txt",
        "sudo pytest",
        "shutdown now",
        "mkfs.ext4 /dev/example",
        "bash -c 'pytest'",
        "git reset --hard",
        "git clean -fd",
    ],
)
def test_reject_destructive_command(policy: CommandPolicy, command: str) -> None:
    with pytest.raises(CommandPolicyError):
        policy.validate(command)


def test_accept_safe_git_command(policy: CommandPolicy) -> None:
    assert policy.validate("git diff -- src") == ("git", "diff", "--", "src")
