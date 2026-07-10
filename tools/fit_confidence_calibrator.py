"""Fit a confidence calibration artifact from committed parse/groundtruth samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confidence_calibration import fit_calibrator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=ROOT / "docs/samples")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    pairs: list[tuple[float, bool]] = []
    for gt_path in sorted(args.samples.glob("*_groundtruth.json")):
        parse_path = gt_path.with_name(gt_path.name.replace("_groundtruth.json", "_parse.json"))
        if not parse_path.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        parsed = json.loads(parse_path.read_text(encoding="utf-8"))
        for truth, pred in zip(gt.get("segments", []), parsed.get("segments", [])):
            if truth.get("speaker") in {"旁白", "未知", "", "群体"}:
                continue
            confidence = float(pred.get("confidence", 1.0))
            acceptable = set(truth.get("acceptable") or [truth.get("speaker", "")])
            pairs.append((confidence, pred.get("speaker") in acceptable))
    artifact = fit_calibrator(pairs, bins=args.bins)
    artifact["source"] = str(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(pairs), "bins": artifact["bins"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
