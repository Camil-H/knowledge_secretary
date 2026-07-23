import pytest

from src.core.registry import Registry


def _registry(label: str = "widget") -> Registry:
    return Registry(label)


# ----- register -----


def test_register_stores_callable_and_returns_fn_unchanged():
    reg = _registry()

    def fn():
        return "ok"

    returned = reg.register("thing")(fn)

    assert returned is fn
    assert reg.get("thing") is fn


# ----- get -----


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["a"],
        ["b", "a"],
    ],
)
def test_get_unknown_name_raises_keyerror_with_label_and_sorted_names(names):
    reg = _registry("widget")
    for name in names:
        reg.register(name)(lambda: None)

    with pytest.raises(KeyError, match="no widget registered") as ei:
        reg.get("missing")

    assert str(sorted(names)) in str(ei.value)


# ----- all -----


def test_all_returns_copy_not_affecting_subsequent_get_or_all():
    reg = _registry()

    def fn():
        return "original"

    reg.register("thing")(fn)

    snapshot = reg.all()
    snapshot["thing"] = "mutated"
    snapshot["extra"] = "also mutated"

    assert reg.get("thing") is fn
    assert reg.all() == {"thing": fn}


# ----- re-registration -----


def test_reregister_same_name_overwrites_last_wins():
    reg = _registry()

    def first():
        return "first"

    def second():
        return "second"

    reg.register("thing")(first)
    reg.register("thing")(second)

    assert reg.get("thing") is second
    assert reg.all() == {"thing": second}
