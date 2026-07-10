"""Portable confidence calibration for speaker attribution outputs.

The artifact is deliberately simple JSON so calibration can be reviewed,
versioned, and rolled back without shipping a training runtime.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _pava(values: list[float], weights: list[int]) -> list[float]:
    blocks: list[list[float | int]] = []
    for value, weight in zip(values, weights):
        blocks.append([value, weight, 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            left, right = blocks[-2], blocks[-1]
            total_weight = int(left[1]) + int(right[1])
            blocks[-2:] = [[(float(left[0]) * int(left[1]) + float(right[0]) * int(right[1])) / total_weight, total_weight, int(left[2]) + int(right[2])]]
    return [float(block[0]) for block in blocks for _ in range(int(block[2]))]


def fit_calibrator(pairs: Iterable[tuple[float, bool]], bins: int = 10) -> dict:
    """Fit monotonic equal-width empirical accuracy bins."""
    grouped: list[list[bool]] = [[] for _ in range(max(2, int(bins)))]
    for raw_score, correct in pairs:
        score = min(0.999999, max(0.0, float(raw_score)))
        grouped[min(len(grouped) - 1, int(score * len(grouped)))].append(bool(correct))
    values = [(sum(items) / len(items) if items else 0.5) for items in grouped]
    weights = [len(items) for items in grouped]
    calibrated = _pava(values, [max(1, weight) for weight in weights])
    return {
        "version": 1,
        "method": "isotonic_equal_width",
        "bins": len(grouped),
        "values": [round(min(1.0, max(0.0, value)), 6) for value in calibrated],
        "counts": weights,
        "samples": sum(weights),
    }


def apply_calibration(score: float, artifact: dict | None) -> float:
    if not artifact or not artifact.get("values"):
        return min(1.0, max(0.0, float(score)))
    values = [float(value) for value in artifact["values"]]
    clipped = min(0.999999, max(0.0, float(score)))
    idx = min(len(values) - 1, int(clipped * len(values)))
    return min(1.0, max(0.0, values[idx]))


def load_calibrator(path: str | Path) -> dict:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("values"), list):
        raise ValueError("invalid confidence calibrator artifact")
    return data


def calibrate_segments(segments: list, artifact: dict) -> int:
    changed = 0
    for segment in segments:
        before = float(segment.confidence or 0.0)
        after = apply_calibration(before, artifact)
        if abs(after - before) > 1e-9:
            segment.confidence = after
            segment.evidence = f"{segment.evidence or ''}; confidence_calibrated"
            changed += 1
    return changed
