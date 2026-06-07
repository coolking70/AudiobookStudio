from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from evaluate_agnes_batch_parse_sample import extract_text
from evaluate_cross_model_parse_review import (
    DEFAULT_SAMPLE,
    DEFAULT_SOURCE,
    bvp_path,
    build_review_config,
    evaluate_segments,
    force_review_targets,
    load_sample,
    parse_text,
)


DEFAULT_OUT = Path(r"I:\code\aitts\omnivoice-reader\outputs\temp_by_date\2026-06-07\cross_review_matrix_eval.json")


def parse_spec(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("spec must be provider:model")
    provider, model = value.split(":", 1)
    provider = provider.strip()
    model = model.strip()
    if provider not in {"agnes", "tokenhub", "local", "openrouter"}:
        raise argparse.ArgumentTypeError(f"unsupported provider: {provider}")
    if not model:
        raise argparse.ArgumentTypeError("model cannot be empty")
    return provider, model


def api_key_for(provider: str) -> str:
    if provider == "agnes":
        return os.getenv("AGNES_API_KEY") or ""
    if provider == "tokenhub":
        return os.getenv("TOKENHUB_API_KEY") or ""
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY") or ""
    return ""


def model_label(spec: tuple[str, str]) -> str:
    return f"{spec[0]}:{spec[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse once per model, then cross-review with multiple reviewer models.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parse-model", action="append", type=parse_spec, required=True)
    parser.add_argument("--review-model", action="append", type=parse_spec, required=True)
    parser.add_argument("--parse-batch-size", type=int, default=8)
    parser.add_argument("--review-batch-size", type=int, default=5)
    parser.add_argument("--parse-max-tokens", type=int, default=5000)
    parser.add_argument("--review-max-tokens", type=int, default=3500)
    args = parser.parse_args()

    sys.path.insert(0, bvp_path())
    from book_voice_parser import route_to_batch_llm

    _, raw_segments, expected = load_sample(args.sample)
    text = extract_text(args.source)
    output: dict[str, Any] = {
        "source": str(args.source),
        "sample": str(args.sample),
        "text_chars": len(text),
        "parse_models": [model_label(spec) for spec in args.parse_model],
        "review_models": [model_label(spec) for spec in args.review_model],
        "parses": [],
        "matrix": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for parse_provider, parse_model in args.parse_model:
        parse_key = api_key_for(parse_provider)
        if parse_provider != "local" and not parse_key:
            raise SystemExit(f"Missing API key for parse provider {parse_provider}")
        parse_started = time.time()
        parsed, parse_seconds = parse_text(
            parse_provider,
            parse_model,
            parse_key,
            text,
            batch_size=args.parse_batch_size,
            max_tokens=args.parse_max_tokens,
        )
        parse_eval = evaluate_segments(parsed.segments, raw_segments, expected)
        parse_record = {
            "provider": parse_provider,
            "model": parse_model,
            "seconds": parse_seconds,
            "parsed_segments": len(parsed.segments),
            "evaluation": parse_eval,
            "stats": parsed.stats,
        }
        output["parses"].append(parse_record)
        print(
            f"PARSE {model_label((parse_provider, parse_model))}: "
            f"{parse_eval['correct']}/{parse_eval['total_cases']} "
            f"acc={parse_eval['accuracy_total']:.3f} seconds={parse_seconds:.1f}",
            flush=True,
        )

        review_input, review_indices = force_review_targets(parsed.segments, parse_eval["cases"])
        for review_provider, review_model in args.review_model:
            review_key = api_key_for(review_provider)
            if review_provider != "local" and not review_key:
                raise SystemExit(f"Missing API key for review provider {review_provider}")
            review_cfg = build_review_config(review_provider, review_model, review_key, args.review_max_tokens)
            review_started = time.time()
            reviewed_segments, review_stats = route_to_batch_llm(
                review_input,
                review_cfg,
                threshold=0.7,
                review_indices=review_indices,
                batch_size=args.review_batch_size,
                narrator="甘织玲奈子",
            )
            review_seconds = time.time() - review_started
            review_eval = evaluate_segments(reviewed_segments, raw_segments, expected)
            before = {case["source_index"]: case for case in parse_eval["cases"]}
            after = {case["source_index"]: case for case in review_eval["cases"]}
            improved = []
            worsened = []
            for idx, before_case in before.items():
                after_case = after.get(idx)
                if not after_case:
                    continue
                if not before_case["ok"] and after_case["ok"]:
                    improved.append(idx)
                elif before_case["ok"] and not after_case["ok"]:
                    worsened.append(idx)
            matrix_record = {
                "parse_provider": parse_provider,
                "parse_model": parse_model,
                "review_provider": review_provider,
                "review_model": review_model,
                "review_seconds": review_seconds,
                "forced_targets": len(review_indices),
                "parse_correct": parse_eval["correct"],
                "parse_accuracy_total": parse_eval["accuracy_total"],
                "final_correct": review_eval["correct"],
                "final_accuracy_total": review_eval["accuracy_total"],
                "final_accuracy_matched": review_eval["accuracy_matched"],
                "improved": improved,
                "worsened": worsened,
                "review_stats": review_stats,
            }
            output["matrix"].append(matrix_record)
            print(
                f"REVIEW {model_label((parse_provider, parse_model))} -> "
                f"{model_label((review_provider, review_model))}: "
                f"{review_eval['correct']}/{review_eval['total_cases']} "
                f"acc={review_eval['accuracy_total']:.3f} "
                f"improved={len(improved)} worsened={len(worsened)} seconds={review_seconds:.1f}",
                flush=True,
            )
            args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

        parse_record["wall_seconds"] = time.time() - parse_started
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = sorted(
        output["matrix"],
        key=lambda item: (item["final_accuracy_total"], -len(item["worsened"])),
        reverse=True,
    )
    output["summary"] = summary
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary[:20], ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
