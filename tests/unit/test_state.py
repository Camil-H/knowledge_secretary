from datetime import UTC, datetime, timedelta

from src.core import state as state_mod
from src.core.models import Item


def _item(item_id: str) -> Item:
    return Item(
        id=item_id, source="s", section="Sec", title="t", url="u", published=datetime.now(UTC)
    )


def test_is_new_and_mark():
    state = {"ids": {}, "kv": {}}
    it = _item("rss:1")
    assert state_mod.is_new(state, it)
    state_mod.mark(state, [it])
    assert not state_mod.is_new(state, it)


def test_mark_ids():
    state = {"ids": {}, "kv": {}}
    state_mod.mark_ids(state, ["a", "b"])
    assert set(state["ids"]) == {"a", "b"}


def test_prune_drops_old_only():
    old = (datetime.now(UTC).date() - timedelta(days=90)).isoformat()
    recent = datetime.now(UTC).date().isoformat()
    state = {"ids": {"old": old, "recent": recent}, "kv": {}}
    state_mod.prune(state, days=60)
    assert "old" not in state["ids"]
    assert "recent" in state["ids"]


def test_kv_roundtrip():
    state = {"ids": {}, "kv": {}}
    assert state_mod.get_kv(state, "x", 7) == 7
    state_mod.set_kv(state, "x", 42)
    assert state_mod.get_kv(state, "x") == 42
