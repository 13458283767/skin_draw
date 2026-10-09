"""Frame-by-frame equivalence check against the delivery pack algorithm.

Runs three deterministic scenarios (random noise, step press/release, startup
convergence with slow tracking) through both ``skin_draw.preprocessing`` and
the delivery pack ``上位机皮肤预处理算法交付包/skin_preprocess.py``. Every
frame and point must agree (NaN == None) within 1e-9; the A-mode negative
clamp is a documented renderer contract (plan section 2), so it is applied to
the reference output before comparing. Prints a JSON report.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "上位机皮肤预处理算法交付包"
sys.path.insert(0, str(PACK_DIR))

try:
    from skin_preprocess import SkinPreprocessor  # delivery pack (untracked reference)
except ImportError as error:
    raise SystemExit(f"未找到交付包对拍参照 {PACK_DIR / 'skin_preprocess.py'}：{error}")

from skin_draw.preprocessing import BlockPreprocessor

TOLERANCE = 1e-9


def contract(value, norm):
    """Renderer contract kept on top of the pack: norm=A clamps d < 0 to 0."""
    return 0.0 if norm == "A" and value < 0 else value


def scenario(name, shape, valid, *, norm, direction, calib_n, fps, r_ref, generate):
    rows, cols = shape
    empty = {index + 1 for index, ok in enumerate(valid.ravel()) if not ok}
    ours = BlockPreprocessor(shape, valid, norm=norm, direction=direction,
                             calib_n=calib_n, fps=fps, r_ref=r_ref)
    pack = SkinPreprocessor(rows, cols, empty=empty, norm=norm, baseline_mode="slow",
                            direction=direction, calib_n=calib_n, fps=fps, r_ref=r_ref)
    assert pack.base[0].upd_n == ours.upd_n, "slow-tracking block length must match"
    frames = generate()
    max_diff = 0.0
    mismatches = []
    for index, frame in enumerate(frames):
        mine = ours.update(frame)
        theirs = pack.update_frame([float(value) for value in frame.ravel()])
        for row in range(rows):
            for col in range(cols):
                value = mine[row, col]
                reference = theirs[row * cols + col]
                if np.isnan(value) != (reference is None):
                    mismatches.append({"frame": index, "row": row, "col": col,
                                       "skin_draw": None if np.isnan(value) else round(float(value), 9),
                                       "delivery_pack": reference})
                elif reference is not None:
                    max_diff = max(max_diff, abs(value - contract(float(reference), norm)))
    max_baseline_diff = 0.0
    mine = ours.baselines
    for row in range(rows):
        for col in range(cols):
            reference = pack.base[row * cols + col].baseline
            if reference is None:
                if not np.isnan(mine[row, col]):
                    mismatches.append({"frame": "final", "row": row, "col": col,
                                       "baseline": True, "skin_draw": float(mine[row, col]),
                                       "delivery_pack": None})
            else:
                max_baseline_diff = max(max_baseline_diff, abs(float(mine[row, col]) - float(reference)))
    return {"name": name, "frames": len(frames), "points": rows * cols,
            "max_abs_diff": max_diff, "max_baseline_diff": max_baseline_diff,
            "mismatches": len(mismatches), "examples": mismatches[:5],
            "ok": not mismatches and max_diff <= TOLERANCE and max_baseline_diff <= TOLERANCE}


def random_scenario():
    shape = (4, 3)
    valid = np.ones(shape, bool)
    valid[1, 2] = False  # arbitrary empty point
    rng = np.random.default_rng(20240521)

    def generate():
        frames = []
        for index in range(420):
            level = 2500.0 if not (200 <= index < 260) else 2240.0
            frames.append(np.rint(level + rng.normal(0, 12, shape)))
        return frames

    return scenario("random-noise", shape, valid, norm="B", direction="drop",
                    calib_n=60, fps=90.6, r_ref=320.0, generate=generate)


def step_scenario():
    shape = (3, 4)
    valid = np.ones(shape, bool)
    valid[0, 0] = False

    def generate():
        frames = []
        for index in range(300):
            if index < 100:
                value = 2500.0
            elif index < 140:
                value = 2600.0
            elif index < 180:
                value = 2350.0
            else:
                value = 2500.0
            frame = np.full(shape, value)
            if index == 210:
                frame[2, 1] = 900.0  # single-frame low rail: M3 must hold
            if index == 211:
                frame[2, 2] = 4095.0
            frames.append(frame)
        return frames

    return scenario("step-press", shape, valid, norm="A", direction="rise",
                    calib_n=45, fps=60.0, r_ref=600.0, generate=generate)


def convergence_scenario():
    shape = (2, 2)
    valid = np.ones(shape, bool)
    rng = np.random.default_rng(7)

    def generate():
        frames = []
        for index in range(620):
            level = 4070.0 if index < 150 else 2500.0  # floating startup platform
            frames.append(np.rint(level + rng.normal(0, 8, shape)))
        return frames

    return scenario("startup-convergence", shape, valid, norm="B", direction="drop",
                    calib_n=90, fps=50.0, r_ref=500.0, generate=generate)


def main():
    scenarios = [random_scenario(), step_scenario(), convergence_scenario()]
    report = {"tolerance": TOLERANCE,
              "note": "norm=A negative deviation is clamped to 0 on both sides (renderer contract)",
              "scenarios": scenarios,
              "ok": all(item["ok"] for item in scenarios)}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["ok"]:
        raise SystemExit("对拍失败：skin_draw 与交付包输出不一致")


if __name__ == "__main__":
    main()
