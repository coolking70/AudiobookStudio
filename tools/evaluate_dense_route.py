"""Evaluate the opt-in dense-scene model route against a frozen sample."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "docs" / "samples"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "BookVoiceParser"))
sys.path.insert(0, str(ROOT / "tools"))

from book_voice_parser.review_router import (  # noqa: E402
    LLMRouterConfig,
    SKIP_SPEAKERS,
    route_dense_to_llm,
    route_to_batch_llm,
)
from book_voice_parser.schema import SegmentEx  # noqa: E402
from run_regression import MANIFEST, score_sample  # noqa: E402


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']?(.*?)["\']?\s*$', line)
        if match:
            os.environ.setdefault(match.group(1), match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="muli4_seg8")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--strategy", choices=["spc", "batch"], default="spc")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv(args.key_env, "")
    if not api_key:
        raise SystemExit(f"missing {args.key_env}")
    parse_path = SAMPLES / f"{args.sample}_parse.json"
    payload = json.loads(parse_path.read_text(encoding="utf-8"))
    segments = [SegmentEx.model_validate(item) for item in payload.get("segments", [])]
    before = score_sample(args.sample, parse_path) or {}
    config = LLMRouterConfig(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=0.0,
        timeout=180,
    )
    started = time.perf_counter()
    if args.strategy == "spc":
        routed, route_stats = route_dense_to_llm(
            segments,
            config,
            threshold=args.threshold,
        )
    else:
        names = [str(segment.speaker or "") for segment in segments]
        targets = sorted({
            index
            for index, name in enumerate(names)
            if name not in SKIP_SPEAKERS
            and len({
                names[j]
                for j in range(max(0, index - 4), min(len(names), index + 5))
                if names[j] not in SKIP_SPEAKERS
            }) >= 3
        })
        routed, route_stats = route_to_batch_llm(
            segments,
            config,
            threshold=args.threshold,
            review_indices=targets,
            batch_size=8,
            narrator=MANIFEST.get(args.sample),
        )
        route_stats.update({"mode": "dense_batch_route", "target_indices": targets})
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "segments": [segment.model_dump(mode="json") for segment in routed],
        "stats": {
            "dense_model_route": route_stats,
            "evaluation": {
                "sample": args.sample,
                "model": args.model,
                "seconds": round(elapsed, 3),
                "before": before,
            },
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    after = score_sample(args.sample, args.output) or {}
    print(json.dumps({
        "sample": args.sample,
        "model": args.model,
        "strategy": args.strategy,
        "seconds": round(elapsed, 3),
        "targets": route_stats.get("targets", 0),
        "corrected": route_stats.get("corrected", 0),
        "blocked": route_stats.get("blocked", 0),
        "failed": route_stats.get("failed", 0),
        "before": before,
        "after": after,
        "output": str(args.output),
    }, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
