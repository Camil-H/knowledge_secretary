import json
from datetime import UTC, datetime, timedelta

import pytest

from src.core import state as state_mod
from src.core.models import Item


def _item(item_id: str) -> Item:
    return Item(
        id=item_id, source="s", section="Sec", title="t", url="u", published=datetime.now(UTC)
    )


def test_is_new_flips_after_mark_ids():
    state = {"ids": {}, "kv": {}}
    it = _item("rss:1")
    assert state_mod.is_new(state, it)
    state_mod.mark_ids(state, [it.id])
    assert not state_mod.is_new(state, it)


def test_mark_ids():
    state = {"ids": {}, "kv": {}}
    state_mod.mark_ids(state, ["a", "b"])
    assert set(state["ids"]) == {"a", "b"}


# ----- prune boundary -----


def test_prune_drops_old_only():
    old = (datetime.now(UTC).date() - timedelta(days=90)).isoformat()
    recent = datetime.now(UTC).date().isoformat()
    state = {"ids": {"old": old, "recent": recent}, "kv": {}}
    state_mod.prune(state, days=60)
    assert "old" not in state["ids"]
    assert "recent" in state["ids"]


def test_prune_keeps_id_dated_exactly_on_cutoff():
    days = 60
    cutoff = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()
    state = {"ids": {"boundary": cutoff}, "kv": {}}
    state_mod.prune(state, days=days)
    assert "boundary" in state["ids"]


def test_kv_roundtrip():
    state = {"ids": {}, "kv": {}}
    assert state_mod.get_kv(state, "x", 7) == 7
    state_mod.set_kv(state, "x", 42)
    assert state_mod.get_kv(state, "x") == 42


# ----- load: missing file -----


def test_load_nonexistent_path_returns_fresh_state(tmp_path):
    path = tmp_path / "missing.json"
    assert not path.exists()
    assert state_mod.load(str(path)) == {"ids": {}, "kv": {}}


# ----- load: existing file missing keys gets them defaulted -----


@pytest.mark.parametrize(
    "on_disk",
    [
        pytest.param({}, id="missing_both"),
        pytest.param({"ids": {"a": "2020-01-01"}}, id="missing_kv"),
        pytest.param({"kv": {"x": 1}}, id="missing_ids"),
    ],
)
def test_load_defaults_missing_keys_via_setdefault(tmp_path, on_disk):
    path = tmp_path / "seen.json"
    path.write_text(json.dumps(on_disk))
    loaded = state_mod.load(str(path))
    assert loaded["ids"] == on_disk.get("ids", {})
    assert loaded["kv"] == on_disk.get("kv", {})


# ----- save+load roundtrip -----


def test_save_creates_nested_parent_dir_and_load_roundtrips(tmp_path):
    path = tmp_path / "nested" / "dir" / "seen.json"
    assert not path.parent.exists()
    original = {"ids": {"a": "2020-01-01"}, "kv": {"k": "v"}}

    state_mod.save(original, str(path))

    assert path.parent.is_dir()
    assert state_mod.load(str(path)) == original
