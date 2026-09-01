import pytest
from incidentlens.parser import parse_line


def test_parse_regular_incident() -> None:
    incident = parse_line("2026-08-31T10:00:00Z|api|INFO|120|ok")

    assert incident.service == "api"
    assert incident.duration_ms == 120
    assert incident.message == "ok"


def test_message_may_contain_pipe_delimiters() -> None:
    incident = parse_line(
        "2026-08-31T10:00:01Z|payments|ERROR|1400|payment|gateway timeout"
    )

    assert incident.message == "payment|gateway timeout"


def test_invalid_field_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 5 fields"):
        parse_line("too|few|fields")
