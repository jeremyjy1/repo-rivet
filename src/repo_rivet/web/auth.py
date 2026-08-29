"""One-use bootstrap authentication and same-origin write protection."""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, Response, status

SESSION_COOKIE = "reporivet_session"


@dataclass(slots=True)
class LocalAuth:
    expected_origin: str
    bootstrap_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    _session_token: str | None = None
    _csrf_token: str | None = None
    _bootstrap_used: bool = False

    def bootstrap(self, token: str, response: Response) -> str:
        if self._bootstrap_used or not hmac.compare_digest(token, self.bootstrap_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired bootstrap token")
        self._bootstrap_used = True
        self._session_token = secrets.token_urlsafe(32)
        self._csrf_token = secrets.token_urlsafe(24)
        response.set_cookie(
            SESSION_COOKIE,
            self._session_token,
            httponly=True,
            samesite="strict",
            secure=self.expected_origin.startswith("https://"),
            path="/",
        )
        return self._csrf_token

    def require_session(self, request: Request) -> None:
        actual = request.cookies.get(SESSION_COOKIE, "")
        if self._session_token is None or not hmac.compare_digest(actual, self._session_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")

    def require_write(self, request: Request) -> None:
        self.require_session(request)
        origin = request.headers.get("origin")
        if origin != self.expected_origin:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid request origin")
        csrf = request.headers.get("x-csrf-token", "")
        if self._csrf_token is None or not hmac.compare_digest(csrf, self._csrf_token):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid CSRF token")

    @property
    def expected_host(self) -> str:
        return self.expected_origin.split("://", maxsplit=1)[-1]
