from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from src.core.models import Context, Item
from src.tasks.youtube import _window_utc, run

TZ = ZoneInfo("America/New_York")


# ----- window math (DST-safe) -----


def test_window_summer_edt():
    now_et = datetime(2026, 7, 15, 15, 0, tzinfo=TZ)  # EDT = UTC-4
    start, end = _window_utc(now_et, {"start": "08:00", "end": "07:59"}, TZ)
    assert start == datetime(2026, 7, 14, 12, 0, tzinfo=UTC)  # 08:00 EDT
    assert end == datetime(2026, 7, 15, 11, 59, 59, tzinfo=UTC)  # 07:59 EDT


def test_window_winter_est():
    now_et = datetime(2026, 1, 15, 15, 0, tzinfo=TZ)  # EST = UTC-5
    start, end = _window_utc(now_et, {"start": "08:00", "end": "07:59"}, TZ)
    assert start == datetime(2026, 1, 14, 13, 0, tzinfo=UTC)  # 08:00 EST
    assert end == datetime(2026, 1, 15, 12, 59, 59, tzinfo=UTC)  # 07:59 EST


# ----- task: window filtering + consumed (the burn-fix) -----


def _cfg():
    return {
        "timezone": "America/New_York",
        "tasks": {
            "youtube": {
                "window_et": {"start": "08:00", "end": "07:59"},
                "sources": [
                    {
                        "key": "yt_x",
                        "kind": "feed",
                        "section": "Pure Science",
                        "handle": "@x",
                        "enrich": ["transcript"],
                    }
                ],
            }
        },
    }


def _et_to_utc(d, hh):
    return datetime.combine(d, time(hh, 0), tzinfo=TZ).astimezone(UTC)


def test_run_keeps_only_in_window_and_reports_consumed():
    today = datetime.now(TZ).date()
    in_item = Item(
        id="yt:IN",
        source="yt_x",
        section="Pure Science",
        title="In Video",
        url="http://y/IN",
        published=_et_to_utc(today - timedelta(days=1), 12),
        text="transcript body",
        meta={"channel": "ChanX"},
    )
    # published today 23:00 ET is always after the 07:59 window end -> excluded, not burned
    out_item = Item(
        id="yt:OUT",
        source="yt_x",
        section="Pure Science",
        title="Out Video",
        url="http://y/OUT",
        published=_et_to_utc(today, 23),
        text="later",
        meta={"channel": "ChanX"},
    )

    ctx = Context(
        cfg=_cfg(),
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: [in_item, out_item],
        call=lambda tier, system, user, max_tokens=None: "- b1\n- b2\n- b3",
        log=lambda m: None,
    )
    result = run(ctx)

    assert result.consumed == ["yt:IN"]  # OUT stays unmarked -> resurfaces next run
    assert "In Video" in result.markdown
    assert "Out Video" not in result.markdown
    assert "- Pure Science" in result.markdown
    assert "- b1" in result.markdown


def test_run_empty_window_yields_blank_markdown():
    ctx = Context(
        cfg=_cfg(),
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: [],
        call=lambda *a, **k: "x",
        log=lambda m: None,
    )
    result = run(ctx)
    assert result.markdown == ""
    assert result.consumed == []
