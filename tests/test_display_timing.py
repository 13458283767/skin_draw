"""The live display throttle is sub-10 ms, so its clock choice is load bearing.

Windows implements time.monotonic() with GetTickCount64, whose granularity is
about 15.6 ms. Subtracting two such samples yields only 0 or ~15.6 ms, which
would quantize the display budget to whole system ticks and silently coarsen
DISPLAY_INTERVAL_MS. These tests pin both the shared-clock invariant and the
observable throttle behaviour.
"""

import time

from skin_draw import live, models
from skin_draw.ui import main_window as mw
from skin_draw.ui.main_window import DISPLAY_INTERVAL_MS, MainWindow


def test_display_path_shares_one_high_resolution_clock():
    # _check_freshness compares main_window's clock against a timestamp stamped
    # by live.py, and gui_sample_ages subtracts snapshot.latest_received_at from
    # a main_window timestamp. Mixing clocks there yields a meaningless age, so
    # every module in this comparison must resolve to the same function.
    assert mw.perf_counter is time.perf_counter
    assert live.perf_counter is time.perf_counter
    assert models.perf_counter is time.perf_counter
    assert mw.perf_counter is not time.monotonic


def fake_clock(monkeypatch, now=100.0):
    """Freeze the module clock so the throttle arithmetic stays deterministic."""
    state = {"now": now}
    monkeypatch.setattr(mw, "perf_counter", lambda: state["now"])
    return state


def ready_window(app, monkeypatch):
    window = MainWindow()
    window.accept_scans = True
    window.worker = object()   # only required to be non-None by the guard
    return window


def test_first_commit_is_immediate(app, monkeypatch):
    clock = fake_clock(monkeypatch)
    window = ready_window(app, monkeypatch)
    try:
        assert window.last_gui_tick is None
        window._request_live_update()
        assert window.display_timer.isActive()
        assert window.display_timer.interval() == 0
    finally:
        clock["now"] += 1
        window.display_timer.stop()
        window.accept_scans = False
        window.worker = None
        window.close()


def test_quiet_gap_commits_immediately_instead_of_waiting(app, monkeypatch):
    clock = fake_clock(monkeypatch)
    window = ready_window(app, monkeypatch)
    try:
        window.last_gui_tick = clock["now"] - (DISPLAY_INTERVAL_MS + 1) / 1000
        window._request_live_update()
        assert window.display_timer.interval() == 0
    finally:
        window.display_timer.stop()
        window.accept_scans = False
        window.worker = None
        window.close()


def test_only_the_unspent_remainder_is_waited(app, monkeypatch):
    clock = fake_clock(monkeypatch)
    window = ready_window(app, monkeypatch)
    try:
        window.last_gui_tick = clock["now"]
        window._request_live_update()
        assert window.display_timer.interval() == DISPLAY_INTERVAL_MS

        # Assert the property, not the exact ceil(): subtracting two floats can
        # land a hair above an integer and ceil() then waits one extra tick.
        window.display_timer.stop()
        window.last_gui_tick = clock["now"] - 1.5 / 1000
        window._request_live_update()
        waited = window.display_timer.interval()
        assert DISPLAY_INTERVAL_MS - 2 <= waited < DISPLAY_INTERVAL_MS
    finally:
        window.display_timer.stop()
        window.accept_scans = False
        window.worker = None
        window.close()


def test_a_burst_does_not_reschedule_the_pending_commit(app, monkeypatch):
    clock = fake_clock(monkeypatch)
    window = ready_window(app, monkeypatch)
    try:
        window.last_gui_tick = clock["now"]
        window._request_live_update()
        scheduled = window.display_timer.remainingTime()
        # Further arrivals inside the window must coalesce onto the pending
        # commit rather than pushing it back, or a busy bus would starve the UI.
        for _ in range(10):
            clock["now"] += 0.0002
            window._request_live_update()
        assert window.display_timer.remainingTime() <= scheduled
    finally:
        window.display_timer.stop()
        window.accept_scans = False
        window.worker = None
        window.close()
