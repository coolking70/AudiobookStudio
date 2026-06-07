from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_SAMPLE = Path(r"I:\code\aitts\omnivoice-reader\docs\samples\muli4_part001_first_hour_bookmark_regression.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _segment_from_snapshot(raw: dict) -> dict:
    item = dict(raw)
    if item.get("cfg_value") == "":
        item["cfg_value"] = None
    if item.get("inference_timesteps") == "":
        item["inference_timesteps"] = None
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Check bookmark-backed role attribution regression cases.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    sys.path.insert(0, str(_repo_root() / "BookVoiceParser"))
    from book_voice_parser.address_term_backcheck import apply_address_term_backcheck
    from book_voice_parser.schema import SegmentEx

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    raw_snapshot = Path(sample["source"]["raw_snapshot"])
    raw_data = json.loads(raw_snapshot.read_text(encoding="utf-8"))
    segments = [SegmentEx.model_validate(_segment_from_snapshot(item)) for item in raw_data["segments"]]
    updated, stats = apply_address_term_backcheck(segments, review_threshold=0.7)

    failures: list[dict] = []
    for case in sample["corrected_segments"]:
        index = int(case["index"])
        expected = str(case["expected_speaker"])
        actual = str(updated[index].speaker)
        if actual != expected:
            failures.append(
                {
                    "index": index,
                    "expected": expected,
                    "actual": actual,
                    "text": updated[index].text[:80],
                    "reason": case.get("reason"),
                }
            )

    print(json.dumps({
        "total": len(sample["corrected_segments"]),
        "passed": len(sample["corrected_segments"]) - len(failures),
        "failed": len(failures),
        "stats": {
            key: stats.get(key)
            for key in [
                "corrected",
                "iterative_corrected",
                "context_corrected",
                "relation_vocative_corrected",
                "blocked",
            ]
        },
        "failures": failures,
    }, ensure_ascii=False, indent=2))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
