from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repo_rivet.web.app import create_app


def _config(path: Path) -> Path:
    path.write_text(
        """
[api]
api_key = "test-secret"
base_url = "https://example.com/v1"
model = "test-model"
context_window_tokens = 32768
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def web_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, str]:
    monkeypatch.setenv("REPORIVET_HOME", str(tmp_path / "home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "hello.py").write_text("print('hello')\n", encoding="utf-8")
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<main>RepoRivet</main>", encoding="utf-8")
    app = create_app(
        workspace=workspace,
        config_path=_config(tmp_path / "reporivet.toml"),
        expected_origin="http://testserver",
        bootstrap_token="bootstrap-token-with-enough-entropy",
        static_dir=static,
    )
    return TestClient(app), "bootstrap-token-with-enough-entropy"


def _authenticate(client: TestClient, token: str) -> str:
    response = client.post("/api/v1/auth/bootstrap", json={"token": token})
    assert response.status_code == 200
    return response.json()["csrf_token"]


def test_bootstrap_is_one_use_and_creates_httponly_cookie(
    web_client: tuple[TestClient, str],
) -> None:
    client, token = web_client

    response = client.post("/api/v1/auth/bootstrap", json={"token": token})

    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    repeated = client.post("/api/v1/auth/bootstrap", json={"token": token})
    assert repeated.status_code == 401


def test_api_requires_session_cookie(web_client: tuple[TestClient, str]) -> None:
    client, _ = web_client

    response = client.get("/api/v1/bootstrap")

    assert response.status_code == 401


def test_writes_require_csrf_and_exact_origin(web_client: tuple[TestClient, str]) -> None:
    client, token = web_client
    csrf = _authenticate(client, token)

    missing_csrf = client.post(
        "/api/v1/sessions",
        headers={"Origin": "http://testserver"},
        json={},
    )
    wrong_origin = client.post(
        "/api/v1/sessions",
        headers={"Origin": "http://evil.test", "X-CSRF-Token": csrf},
        json={},
    )
    accepted = client.post(
        "/api/v1/sessions",
        headers={"Origin": "http://testserver", "X-CSRF-Token": csrf},
        json={"name": "GUI session"},
    )

    assert missing_csrf.status_code == 403
    assert wrong_origin.status_code == 403
    assert accepted.status_code == 200
    assert accepted.json()["name"] == "GUI session"


def test_workspace_file_endpoint_rejects_path_traversal(
    web_client: tuple[TestClient, str],
) -> None:
    client, token = web_client
    _authenticate(client, token)

    response = client.get("/api/v1/workspace/file", params={"path": "../secret.txt"})

    assert response.status_code == 409
    assert "escapes workspace" in response.json()["detail"]


def test_security_headers_are_applied(web_client: tuple[TestClient, str]) -> None:
    client, _ = web_client

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "object-src 'none'" in response.headers["content-security-policy"]


def test_skills_endpoint_uses_manifest_summary_as_description(
    web_client: tuple[TestClient, str],
) -> None:
    client, token = web_client
    _authenticate(client, token)

    response = client.get("/api/v1/skills")

    assert response.status_code == 200
    skills = response.json()
    assert skills
    assert all(item["description"] for item in skills)
    assert {item["source"] for item in skills} == {"system"}
