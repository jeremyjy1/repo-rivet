"""Session-management failures exposed by the CLI."""


class SessionError(Exception):
    """Base class for expected session failures."""


class SessionNotFound(SessionError):
    """No session matched the supplied identifier."""


class AmbiguousSessionId(SessionError):
    """A short identifier matched more than one session."""


class SessionAlreadyRunning(SessionError):
    """Another live process owns the session lock."""


class SessionNotResumable(SessionError):
    """The session lifecycle does not permit direct continuation."""


class SessionWorkspaceMismatch(SessionError):
    """A session was selected from a different workspace."""


class SessionCorrupted(SessionError):
    """Persisted session data cannot be validated."""


class SessionLockStale(SessionError):
    """A dead process appears to have left a lock behind."""
