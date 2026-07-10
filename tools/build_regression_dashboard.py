"""Build a dependency-free regression dashboard from committed samples."""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_regression import SAMP, canon, is_crowd, norm, pick_segs, scene_size, score_sample  # noqa: E402


def build_dashboard(parse_dir: Path, sample_dir: Path = SAMP) -> dict:
    rows = []
    confidence: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    errors: dict[str, int] = defaultdict(int)
    for seg_name in sorted(pick_segs(None)):
        gt_path = sample_dir / f"{seg_name}_groundtruth.json"
        parse_path = parse_dir / f"{seg_name}_parse.json"
        scored = score_sample(seg_name, parse_dir / parse_path.name)
        if not scored:
            continue
        rows.append(scored)
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        parsed = json.loads(parse_path.read_text(encoding="utf-8"))
        for truth, pred in zip(gt.get("segments", []), parsed.get("segments", [])):
            if is_crowd(str(truth.get("speaker", ""))) or norm(str(truth.get("text", ""))) != norm(str(pred.get("text", ""))):
                continue
            try:
                score = float(pred.get("confidence", 1.0))
            except (TypeError, ValueError):
                score = 0.0
            bucket = "<0.50" if score < 0.5 else "0.50-0.70" if score < 0.7 else "0.70-0.85" if score < 0.85 else ">=0.85"
            confidence[bucket][1] += 1
            acceptable = set(truth.get("acceptable") or [truth.get("speaker", "")])
            ok = canon(str(pred.get("speaker", ""))) in {canon(x) for x in acceptable}
            confidence[bucket][0] += int(ok)
            if not ok:
                dense = scene_size([str(item.get("speaker", "")) for item in gt.get("segments", [])], int(truth.get("i", 0))) >= 3
                errors["dense_scene" if dense else "simple_scene"] += 1
                errors[str(pred.get("attribution_type") or "unknown")] += 1
    total = sum(row["named"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parse_dir": str(parse_dir),
        "overall": {"named": total, "correct": correct, "accuracy": round(correct / total, 6) if total else None},
        "samples": rows,
        "confidence_buckets": {
            key: {"correct": value[0], "count": value[1], "accuracy": round(value[0] / value[1], 6) if value[1] else None}
            for key, value in sorted(confidence.items())
        },
        "error_categories": dict(sorted(errors.items(), key=lambda item: (-item[1], item[0]))),
    }


def render_html(data: dict) -> str:
    overall = data["overall"]
    rows = "".join(f"<tr><td>{html.escape(row['seg'])}</td><td>{row['named']}</td><td>{row['acc']:.2%}</td><td>{row['dense'][0]}/{row['dense'][1]}</td></tr>" for row in data["samples"])
    buckets = "".join(f"<tr><td>{html.escape(key)}</td><td>{value['count']}</td><td>{value['accuracy']:.2%}</td></tr>" for key, value in data["confidence_buckets"].items() if value["accuracy"] is not None)
    errors = "".join(f"<li>{html.escape(key)}: {value}</li>" for key, value in data["error_categories"].items())
    return f"<!doctype html><meta charset='utf-8'><title>Regression dashboard</title><h1>Regression dashboard</h1><p>Overall: {overall['correct']}/{overall['named']} ({overall['accuracy']:.2%})</p><h2>Samples</h2><table><tr><th>Sample</th><th>Named</th><th>Accuracy</th><th>Dense</th></tr>{rows}</table><h2>Confidence calibration</h2><table><tr><th>Bucket</th><th>Count</th><th>Accuracy</th></tr>{buckets}</table><h2>Error categories</h2><ul>{errors}</ul>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse-dir", type=Path, default=SAMP)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = build_dashboard(args.parse_dir, SAMP)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".html").write_text(render_html(data), encoding="utf-8")
    print(f"dashboard: {data['overall']['correct']}/{data['overall']['named']} -> {args.output}")


if __name__ == "__main__":
    main()
