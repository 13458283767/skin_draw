"""Raise the Windows system timer resolution so Qt timers hit ~1 ms.

Windows quantizes timer callbacks to the system timer resolution, which is
about 15.625 ms by default. QTimer(Qt.PreciseTimer) still runs on that coarse
clock unless the process asks for a finer one through timeBeginPeriod, so the
8 ms display throttle can actually wait 15 ms or more. Raising the resolution
makes the existing PreciseTimer meaningful at the cost of slightly higher
power draw, which is acceptable for a measurement workstation.

Every call must be paired: timeBeginPeriod/timeEndPeriod are reference
counted per process. enable_high_resolution_timer() and
restore_timer_resolution() keep that balance internally, so a stray second
caller cannot release the period early.
"""

import logging
import sys

logger = logging.getLogger(__name__)

MILLISECONDS = 1
_enabled = False


def enable_high_resolution_timer() -> bool:
    """Request 1 ms timer resolution; return True when it is in effect."""
    global _enabled
    if _enabled:
        return True
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        result = ctypes.windll.winmm.timeBeginPeriod(MILLISECONDS)
        if result != 0:
            logger.warning("timeBeginPeriod(%d) 返回错误码 %s", MILLISECONDS, result)
            return False
        _enabled = True
        logger.info("Windows 定时器精度已提升到 %d ms", MILLISECONDS)
        return True
    except Exception as error:  # ctypes/winmm unavailable, never fatal to start
        logger.warning("提升 Windows 定时器精度失败：%s", error)
        return False


def restore_timer_resolution() -> None:
    """Release the 1 ms request. Safe to call repeatedly or without enable."""
    global _enabled
    if not _enabled:
        return
    _enabled = False
    try:
        import ctypes

        ctypes.windll.winmm.timeEndPeriod(MILLISECONDS)
        logger.info("已恢复 Windows 默认定时器精度")
    except Exception as error:
        logger.warning("恢复 Windows 定时器精度失败：%s", error)
