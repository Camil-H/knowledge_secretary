# tests/unit/fetchers/conftest.py
"""Doubles and fixtures shared by the fetcher test modules.

The url-guard relaxation is offered here as a plain fixture, not an autouse one:
each guarded fetcher's module opts in for its own module only, so a future test
that should exercise the guard cannot be silently disarmed from here.
"""

from types import ModuleType
from typing import Any

import pytest

# == Test doubles =============================================================


class _FakeResp:
    """A JSON API response; only `.json()` is ever read."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _BadJsonResp:
    def json(self):
        raise ValueError("not json")


class _FakeHttpResp:
    def __init__(self, content=b"", status_code=200, text=""):
        self.content = content
        self.status_code = status_code
        self.text = text


# == Fixtures =================================================================


@pytest.fixture
def _allow_every_url(monkeypatch: pytest.MonkeyPatch):
    """Hand back a callable that neutralizes one fetcher module's url guard."""

    def _allow(module: ModuleType) -> None:
        monkeypatch.setattr(module, "assert_safe_url", lambda _u: None)

    return _allow


# == Helper Functions =========================================================


def _raiser(exc: BaseException):
    def _raise(*_a: Any, **_k: Any):
        raise exc

    return _raise
