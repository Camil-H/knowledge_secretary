import pytest

from src.core.errors import AuthError, ExternalError


@pytest.mark.parametrize(
    ("detail", "cause", "expected"),
    [
        ("bad token", ValueError("underlying"), "src: bad token"),
        ("", ValueError("underlying"), "src: underlying"),
        ("", None, "src"),
    ],
)
def test_external_error_message_composition(detail: str, cause: Exception | None, expected: str):
    err = ExternalError("src", cause=cause, detail=detail)
    assert str(err) == expected


def test_external_error_preserves_source_and_cause():
    cause = ValueError("boom")
    err = ExternalError("rss", cause=cause, detail="fetch failed")
    assert err.source == "rss"
    assert err.cause is cause


def test_external_error_cause_defaults_to_none():
    err = ExternalError("rss", detail="fetch failed")
    assert err.cause is None


def test_auth_error_is_external_error_subclass():
    assert issubclass(AuthError, ExternalError)


def test_auth_error_caught_by_generic_external_error_handler():
    with pytest.raises(ExternalError):
        raise AuthError("llm", detail="invalid api key")
