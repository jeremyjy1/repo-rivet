import json
from pathlib import Path

from repo_rivet.storage.event_logger import EventLogger, create_session_logger


def test_event_logger_writes_jsonl_and_redacts_secrets(tmp_path: Path) -> None:
    secret = "super-secret-value"
    log_path = tmp_path / "logs" / "session.jsonl"
    logger = EventLogger(log_path, secrets=(secret,))

    logger.log(
        "tool_call",
        api_key=secret,
        arguments={"content": f"prefix {secret} suffix"},
    )

    raw_line = log_path.read_text(encoding="utf-8")
    event = json.loads(raw_line)
    assert event["event"] == "tool_call"
    assert event["data"]["api_key"] == "[REDACTED]"
    assert event["data"]["arguments"]["content"] == "[REDACTED]"
    assert secret not in raw_line


def test_event_logger_redacts_inline_authorization_values(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    logger = EventLogger(log_path)

    logger.log("command", command="curl -H 'Authorization: Bearer another-secret' example.com")

    raw_line = log_path.read_text(encoding="utf-8")
    assert "another-secret" not in raw_line
    assert "[REDACTED]" in raw_line


def test_event_logger_keeps_non_secret_token_and_content_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    logger = EventLogger(log_path)

    logger.log(
        "model_usage",
        raw_estimated_prompt_tokens=12_345,
        completion_tokens=678,
        content_length=900,
        access_token="must-not-leak",
        content="private file contents",
    )

    data = json.loads(log_path.read_text(encoding="utf-8"))["data"]
    assert data["raw_estimated_prompt_tokens"] == 12_345
    assert data["completion_tokens"] == 678
    assert data["content_length"] == 900
    assert data["access_token"] == "[REDACTED]"
    assert data["content"] == "[REDACTED]"


def test_create_session_logger_uses_jsonl_file(tmp_path: Path) -> None:
    logger = create_session_logger(tmp_path)

    logger.log("session_start", task="task")

    assert logger.path.parent == tmp_path
    assert logger.path.suffix == ".jsonl"
    assert logger.path.exists()
