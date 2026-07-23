"""Shared error types for external dependencies (source fetchers + the LLM client).

A failure a caller may tolerate is raised as ExternalError (source label + cause)
and degraded at the source's own boundary. AuthError is the one a caller must NOT
degrade — a credential failure propagates so it fails loudly.
"""


class ExternalError(Exception):
    """A failure talking to an external dependency (source name + underlying cause)."""

    def __init__(self, source: str, *, cause: Exception | None = None, detail: str = "") -> None:
        self.source = source
        self.cause = cause
        message = detail or (str(cause) if cause else "")
        super().__init__(f"{source}: {message}" if message else source)


class AuthError(ExternalError):
    """A credential / authorization failure — propagated, never degraded."""
