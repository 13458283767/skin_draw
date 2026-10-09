import ctypes
import sys
import types

import pytest

from skin_draw import timer_resolution


@pytest.fixture(autouse=True)
def release_timer_period():
    """Never leak a process-wide timer request into other tests.

    This tears down after monkeypatch has restored the real ctypes module, so a
    test that used the real API still gets its request released.
    """
    yield
    timer_resolution.restore_timer_resolution()


def install_fake_winmm(monkeypatch, begin, end=None):
    """Force the Windows branch and capture the exact OS calls issued."""
    winmm = types.SimpleNamespace(timeBeginPeriod=begin)
    if end is not None:
        winmm.timeEndPeriod = end
    monkeypatch.setattr(timer_resolution, "_enabled", False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(winmm=winmm), raising=False)


def test_issues_exactly_one_balanced_pair_and_is_idempotent(monkeypatch):
    calls = []
    install_fake_winmm(monkeypatch,
                       lambda ms: calls.append(("begin", ms)) or 0,
                       lambda ms: calls.append(("end", ms)) or 0)

    assert timer_resolution.enable_high_resolution_timer() is True
    assert timer_resolution.enable_high_resolution_timer() is True
    timer_resolution.restore_timer_resolution()
    timer_resolution.restore_timer_resolution()

    # Two enables must not become two requests, or a single restore would leave
    # the period raised for the remaining life of the process.
    assert calls == [("begin", 1), ("end", 1)]
    assert timer_resolution._enabled is False


def test_restore_is_a_noop_before_any_enable(monkeypatch):
    calls = []
    install_fake_winmm(monkeypatch,
                       lambda ms: calls.append(("begin", ms)) or 0,
                       lambda ms: calls.append(("end", ms)) or 0)

    assert timer_resolution.restore_timer_resolution() is None
    assert calls == []


def test_error_return_is_reported_without_enabling(monkeypatch):
    install_fake_winmm(monkeypatch, lambda ms: 1)  # TIMERR_NOCANDO
    assert timer_resolution.enable_high_resolution_timer() is False
    assert timer_resolution._enabled is False


def test_missing_winmm_is_not_fatal(monkeypatch):
    monkeypatch.setattr(timer_resolution, "_enabled", False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(), raising=False)
    assert timer_resolution.enable_high_resolution_timer() is False


def test_non_windows_never_touches_the_api(monkeypatch):
    calls = []
    monkeypatch.setattr(timer_resolution, "_enabled", False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "windll",
                        types.SimpleNamespace(winmm=types.SimpleNamespace(
                            timeBeginPeriod=lambda ms: calls.append(ms) or 0)),
                        raising=False)
    assert timer_resolution.enable_high_resolution_timer() is False
    assert calls == []


def test_windows_requests_one_millisecond_for_real():
    if sys.platform != "win32":
        pytest.skip("仅 Windows 提供系统定时器精度接口")
    assert timer_resolution.enable_high_resolution_timer() is True
    assert timer_resolution._enabled is True
    assert timer_resolution.restore_timer_resolution() is None
    assert timer_resolution._enabled is False
