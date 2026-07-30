# tests/unit/fetchers/test_x.py
"""Deterministic x fetcher logic. The twitter CLI subprocess is stubbed."""

import json
import logging
import subprocess

import pytest

from src.core.errors import AuthError
from src.fetchers import x
from tests.unit.fetchers.conftest import _raiser

# == x.recent_tweets ==========================================================


def test_recent_tweets_composes_argv_and_parses_stdout(monkeypatch):
    """The handle arrives with a leading "@", so argv also pins the normalization."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return type("Proc", (), {"stdout": json.dumps({"tweets": [{"id": 1}, {"id": 2}]})})()

    monkeypatch.setattr(x.subprocess, "run", fake_run)
    out = x.recent_tweets("@someuser", limit=5)

    assert captured["argv"] == ["twitter", "user-posts", "someuser", "--max", "5", "--json"]
    assert captured["kwargs"]["check"] is True
    assert out == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("handle", ["--json", "-rf", "; rm -rf /", "way-too-long-for-a-handle"])
def test_recent_tweets_rejects_flag_shaped_or_invalid_handle(monkeypatch, caplog, handle):
    def _must_not_run(*_a, **_k):
        raise AssertionError("subprocess must not run for an invalid handle")

    monkeypatch.setattr(x.subprocess, "run", _must_not_run)
    with caplog.at_level(logging.WARNING):
        out = x.recent_tweets(handle)
    assert out == []
    assert any("degraded" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "fake_run",
    [
        _raiser(subprocess.SubprocessError("cli crashed")),
        lambda *a, **k: type("Proc", (), {"stdout": "not json"})(),  # -> JSONDecodeError
    ],
)
def test_recent_tweets_degrades_on_subprocess_or_json_error(monkeypatch, caplog, fake_run):
    monkeypatch.setattr(x.subprocess, "run", fake_run)
    with caplog.at_level(logging.WARNING):
        out = x.recent_tweets("someuser")
    assert out == []
    assert any("degraded" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "stderr",
    [
        pytest.param("Error: 401 Unauthorized — session expired", id="401_unauthorized"),
        pytest.param("HTTP 403 forbidden", id="403_forbidden"),
        pytest.param("invalid api key", id="invalid_api_key"),
    ],
)
def test_recent_tweets_raises_auth_error_on_expired_cookies(monkeypatch, stderr):
    """Each auth marker class reaches the caller as an AuthError telling it what to do."""
    err = subprocess.CalledProcessError(1, "twitter", stderr=stderr)
    monkeypatch.setattr(x.subprocess, "run", _raiser(err))
    with pytest.raises(AuthError, match="renew"):
        x.recent_tweets("someuser")


@pytest.mark.parametrize(
    "stderr",
    [
        pytest.param("temporary network failure", id="network_failure"),
        pytest.param("Error fetching tweets by author handle", id="author_is_not_a_marker"),
        pytest.param("session timed out, expired connection", id="generic_session_expired"),
        pytest.param("request id 14012 failed", id="digit_substring_401"),
        pytest.param(None, id="no_stderr"),
    ],
)
def test_recent_tweets_degrades_on_non_auth_called_process_error(monkeypatch, caplog, stderr):
    """A CLI failure carrying no auth marker degrades to [] instead of raising AuthError."""
    err = subprocess.CalledProcessError(1, "twitter", stderr=stderr)
    monkeypatch.setattr(x.subprocess, "run", _raiser(err))
    with caplog.at_level(logging.WARNING):
        out = x.recent_tweets("someuser")
    assert out == []
    assert any("degraded" in r.message for r in caplog.records)


def test_recent_tweets_propagates_unexpected_format(monkeypatch):
    monkeypatch.setattr(
        x.subprocess,
        "run",
        lambda *a, **k: type("Proc", (), {"stdout": json.dumps({"nope": 5})})(),
    )
    with pytest.raises(x.UnexpectedXFormat):
        x.recent_tweets("someuser")


# == Helper Functions =========================================================

# ----- x._extract -----


@pytest.mark.parametrize(
    ("data", "expected_len"),
    [
        pytest.param([{"id": 1}], 1, id="top_level_array"),
        pytest.param({"tweets": [{"id": 1}, {"id": 2}]}, 2, id="first_wrapper_key"),
        pytest.param({"data": [{"id": 1}]}, 1, id="later_wrapper_key"),
        pytest.param({"nope": 5}, None, id="dict_without_a_list_raises"),
        pytest.param("garbage", None, id="non_container_raises"),
    ],
)
def test_x_extract_reads_known_shapes_or_raises(data, expected_len):
    """`expected_len is None` marks a shape `_extract` must reject rather than unwrap."""
    if expected_len is None:
        with pytest.raises(x.UnexpectedXFormat):
            x._extract(data)
    else:
        assert len(x._extract(data)) == expected_len
