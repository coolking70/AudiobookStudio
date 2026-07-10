"""Fit a confidence calibration artifact from committed parse/groundtruth samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confidence_calibration import calibration_metrics, fit_calibrator  # noqa: E402


def load_sample_pairs(gt_path: Path) -> list[tuple[float, bool]]:
    parse_path = gt_path.with_name(gt_path.name.replace("_groundtruth.json", "_parse.json"))
    if not parse_path.exists():
        return []
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    parsed = json.loads(parse_path.read_text(encoding="utf-8"))
    pairs: list[tuple[float, bool]] = []
    for truth, pred in zip(gt.get("segments", []), parsed.get("segments", [])):
        if truth.get("speaker") in {"旁白", "未知", "", "群体"}:
            continue
        confidence = float(pred.get("confidence", 1.0))
        acceptable = set(truth.get("acceptable") or [truth.get("speaker", "")])
        pairs.append((confidence, pred.get("speaker") in acceptable))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=ROOT / "docs/samples")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--validation-ratio", type=float, default=0.3)
    args = parser.parse_args()
    sample_paths = [path for path in sorted(args.samples.glob("*_groundtruth.json")) if load_sample_pairs(path)]
    validation_count = max(1, round(len(sample_paths) * min(0.8, max(0.1, args.validation_ratio))))
    train_paths = sample_paths[:-validation_count]
    validation_paths = sample_paths[-validation_count:]
    train_pairs = [pair for path in train_paths for pair in load_sample_pairs(path)]
    validation_pairs = [pair for path in validation_paths for pair in load_sample_pairs(path)]
    artifact = fit_calibrator(train_pairs, bins=args.bins)
    try:
        artifact["source"] = str(args.samples.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        artifact["source"] = str(args.samples)
    raw_metrics = calibration_metrics(validation_pairs, bins=args.bins)
    calibrated_metrics = calibration_metrics(validation_pairs, artifact, bins=args.bins)
    artifact["validation"] = {
        "samples": [path.stem.removesuffix("_groundtruth") for path in validation_paths],
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
    }
    artifact["promoted"] = bool(
        validation_pairs
        and calibrated_metrics["brier"] <= raw_metrics["brier"]
        and calibrated_metrics["ece"] <= raw_metrics["ece"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "train_samples": len(train_pairs),
        "validation_samples": len(validation_pairs),
        "bins": artifact["bins"],
        "promoted": artifact["promoted"],
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
