from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from evaluate_agnes_batch_parse_sample import extract_text, normalize_text_for_match
from evaluate_agnes_bookmark_review import ROLE_HINTS


DEFAULT_SOURCE = Path(r"C:\Users\coolking\Downloads\125697 utf-8.txt")
DEFAULT_SAMPLE = Path(r"I:\code\aitts\omnivoice-reader\docs\samples\muli4_part001_first_hour_bookmark_regression.json")
DEFAULT_OUT = Path(r"I:\code\aitts\omnivoice-reader\outputs\temp_by_date\2026-06-07\cross_model_parse_review_eval.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bvp_path() -> str:
    return str(repo_root() / "BookVoiceParser")


def load_sample(sample_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, str]]:
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    raw_segments = json.loads(Path(sample["source"]["raw_snapshot"]).read_text(encoding="utf-8"))["segments"]
    expected = {int(case["index"]): str(case["expected_speaker"]) for case in sample["corrected_segments"]}
    return sample, raw_segments, expected


def build_batch_config(provider: str, model: str, api_key: str, batch_size: int, max_tokens: int):
    sys.path.insert(0, bvp_path())
    from book_voice_parser import BatchConfig

    if provider == "agnes":
        return BatchConfig(
            base_url="https://apihub.agnes-ai.com/v1",
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=240,
            context_chars=320,
            output_mode="compact",
            disable_thinking=True,
        )
    if provider == "local":
        return BatchConfig(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio",
            model=model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=240,
            context_chars=320,
            output_mode="compact",
            disable_thinking=True,
        )
    if provider == "tokenhub":
        return BatchConfig(
            base_url="https://tokenhub.tencentmaas.com/v1",
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=240,
            context_chars=320,
            output_mode="compact",
            disable_thinking=True,
        )
    if provider == "openrouter":
        return BatchConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=300,
            context_chars=320,
            output_mode="compact",
            disable_thinking=True,
        )
    raise ValueError(f"unknown provider: {provider}")


def build_review_config(provider: str, model: str, api_key: str, max_tokens: int):
    sys.path.insert(0, bvp_path())
    from book_voice_parser import LLMRouterConfig

    if provider == "agnes":
        return LLMRouterConfig(
            base_url="https://apihub.agnes-ai.com/v1/chat/completions",
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=240,
        )
    if provider == "local":
        return LLMRouterConfig(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio",
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=240,
        )
    if provider == "tokenhub":
        return LLMRouterConfig(
            base_url="https://tokenhub.tencentmaas.com/v1/chat/completions",
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=240,
        )
    if provider == "openrouter":
        return LLMRouterConfig(
            base_url="https://openrouter.ai/api/v1/chat/completions",
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=300,
        )
    raise ValueError(f"unknown provider: {provider}")


def parse_text(provider: str, model: str, api_key: str, text: str, batch_size: int, max_tokens: int):
    sys.path.insert(0, bvp_path())
    from book_voice_parser import parse_novel

    cfg = build_batch_config(provider, model, api_key, batch_size, max_tokens)
    started = time.time()
    result = parse_novel(
        text,
        role_hints=ROLE_HINTS,
        batch_llm_config=cfg,
        narrator="甘织玲奈子",
        return_result=True,
        include_narration=False,
        review_threshold=0.7,
    )
    return result, time.time() - started


def evaluate_segments(segments: list[Any], raw_segments: list[dict[str, Any]], expected: dict[int, str]) -> dict[str, Any]:
    parsed_by_text: dict[str, list[tuple[int, Any]]] = {}
    for idx, seg in enumerate(segments):
        text = getattr(seg, "text", None)
        if text is None and isinstance(seg, dict):
            text = seg.get("text", "")
        parsed_by_text.setdefault(normalize_text_for_match(str(text or "")), []).append((idx, seg))

    cases: list[dict[str, Any]] = []
    correct = 0
    matched = 0
    for source_index, expected_speaker in expected.items():
        raw_text = str(raw_segments[source_index].get("text") or "")
        matches = parsed_by_text.get(normalize_text_for_match(raw_text)) or []
        if not matches:
            cases.append({
                "source_index": source_index,
                "expected": expected_speaker,
                "speaker": None,
                "ok": False,
                "missing": True,
                "text": raw_text,
            })
            continue
        matched += 1
        parsed_index, seg = matches[0]
        speaker = getattr(seg, "speaker", None)
        confidence = getattr(seg, "confidence", None)
        evidence = getattr(seg, "evidence", None)
        if isinstance(seg, dict):
            speaker = seg.get("speaker")
            confidence = seg.get("confidence")
            evidence = seg.get("evidence")
        ok = str(speaker or "") == expected_speaker
        correct += int(ok)
        cases.append({
            "source_index": source_index,
            "parsed_index": parsed_index,
            "expected": expected_speaker,
            "speaker": str(speaker or ""),
            "ok": ok,
            "confidence": confidence,
            "evidence": evidence,
            "text": raw_text,
        })
    return {
        "total_cases": len(expected),
        "matched_cases": matched,
        "correct": correct,
        "accuracy_total": correct / max(1, len(expected)),
        "accuracy_matched": correct / max(1, matched),
        "cases": cases,
    }


def force_review_targets(segments: list[Any], eval_cases: list[dict[str, Any]]) -> tuple[list[Any], list[int]]:
    copied = [seg.model_copy() for seg in segments]
    indices: list[int] = []
    for case in eval_cases:
        if case.get("missing"):
            continue
        idx = case.get("parsed_index")
        if idx is None:
            continue
        idx = int(idx)
        indices.append(idx)
        copied[idx].confidence = min(float(copied[idx].confidence or 0.5), 0.45)
        copied[idx].evidence = f"{copied[idx].evidence or ''}；cross-model受控复核目标"
    return copied, indices


def run_chain(
    name: str,
    parse_provider: str,
    parse_model: str,
    parse_api_key: str,
    review_provider: str,
    review_model: str,
    review_api_key: str,
    text: str,
    raw_segments: list[dict[str, Any]],
    expected: dict[int, str],
    parse_batch_size: int,
    parse_max_tokens: int,
    review_batch_size: int,
    review_max_tokens: int,
) -> dict[str, Any]:
    sys.path.insert(0, bvp_path())
    from book_voice_parser import route_to_batch_llm

    parsed, parse_seconds = parse_text(
        parse_provider,
        parse_model,
        parse_api_key,
        text,
        batch_size=parse_batch_size,
        max_tokens=parse_max_tokens,
    )
    initial_eval = evaluate_segments(parsed.segments, raw_segments, expected)
    review_input, review_indices = force_review_targets(parsed.segments, initial_eval["cases"])
    review_cfg = build_review_config(review_provider, review_model, review_api_key, review_max_tokens)
    review_started = time.time()
    reviewed_segments, review_stats = route_to_batch_llm(
        review_input,
        review_cfg,
        threshold=0.7,
        review_indices=review_indices,
        batch_size=review_batch_size,
        narrator="甘织玲奈子",
    )
    review_seconds = time.time() - review_started
    reviewed_eval = evaluate_segments(reviewed_segments, raw_segments, expected)
    return {
        "name": name,
        "parse": {
            "provider": parse_provider,
            "model": parse_model,
            "seconds": parse_seconds,
            "parsed_segments": len(parsed.segments),
            "stats": parsed.stats,
            "evaluation": initial_eval,
        },
        "review": {
            "provider": review_provider,
            "model": review_model,
            "seconds": review_seconds,
            "forced_targets": len(review_indices),
            "stats": review_stats,
            "evaluation": reviewed_eval,
        },
    }


def compact_chain(chain: dict[str, Any]) -> dict[str, Any]:
    p = chain["parse"]["evaluation"]
    r = chain["review"]["evaluation"]
    return {
        "chain": chain["name"],
        "parse_model": chain["parse"]["model"],
        "review_model": chain["review"]["model"],
        "parse_correct": p["correct"],
        "parse_total_accuracy": p["accuracy_total"],
        "parse_matched_accuracy": p["accuracy_matched"],
        "review_final_correct": r["correct"],
        "review_final_total_accuracy": r["accuracy_total"],
        "review_final_matched_accuracy": r["accuracy_matched"],
        "matched_cases": p["matched_cases"],
        "review_stats": chain["review"]["stats"],
        "parse_seconds": chain["parse"]["seconds"],
        "review_seconds": chain["review"]["seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local/Agnes parse and cross-model batch review on bookmark regression sample.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--local-model", default="google/gemma-4-12b")
    parser.add_argument("--agnes-model", default="agnes-2.0-flash")
    parser.add_argument("--tokenhub-model", default="deepseek-v4-flash")
    parser.add_argument("--openrouter-model", default="nvidia/nemotron-3-super-120b-a12b:free")
    parser.add_argument("--parse-batch-size", type=int, default=8)
    parser.add_argument("--review-batch-size", type=int, default=5)
    parser.add_argument("--local-parse-max-tokens", type=int, default=5000)
    parser.add_argument("--agnes-parse-max-tokens", type=int, default=5000)
    parser.add_argument("--local-review-max-tokens", type=int, default=3000)
    parser.add_argument("--agnes-review-max-tokens", type=int, default=3000)
    parser.add_argument(
        "--only",
        choices=[
            "local-agnes",
            "agnes-local",
            "tokenhub-agnes",
            "agnes-tokenhub",
            "openrouter-agnes",
            "agnes-openrouter",
            "both",
            "tokenhub-both",
            "openrouter-both",
        ],
        default="both",
    )
    args = parser.parse_args()

    agnes_api_key = os.getenv("AGNES_API_KEY")
    tokenhub_api_key = os.getenv("TOKENHUB_API_KEY")
    if args.only in {"local-agnes", "agnes-local", "both"} and not agnes_api_key:
        raise SystemExit("Please set AGNES_API_KEY in the environment.")
    if args.only in {"tokenhub-agnes", "agnes-tokenhub", "tokenhub-both"} and not tokenhub_api_key:
        raise SystemExit("Please set TOKENHUB_API_KEY in the environment.")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if args.only in {"openrouter-agnes", "agnes-openrouter", "openrouter-both"} and not openrouter_api_key:
        raise SystemExit("Please set OPENROUTER_API_KEY in the environment.")

    sample, raw_segments, expected = load_sample(args.sample)
    text = extract_text(args.source)
    chains: list[dict[str, Any]] = []
    if args.only in {"local-agnes", "both"}:
        chains.append(run_chain(
            "本地初析 -> Agnes复核",
            "local",
            args.local_model,
            "",
            "agnes",
            args.agnes_model,
            agnes_api_key or "",
            text,
            raw_segments,
            expected,
            args.parse_batch_size,
            args.local_parse_max_tokens,
            args.review_batch_size,
            args.agnes_review_max_tokens,
        ))
    if args.only in {"agnes-local", "both"}:
        chains.append(run_chain(
            "Agnes初析 -> 本地复核",
            "agnes",
            args.agnes_model,
            agnes_api_key or "",
            "local",
            args.local_model,
            "",
            text,
            raw_segments,
            expected,
            args.parse_batch_size,
            args.agnes_parse_max_tokens,
            args.review_batch_size,
            args.local_review_max_tokens,
        ))
    if args.only in {"tokenhub-agnes", "tokenhub-both"}:
        chains.append(run_chain(
            "TokenHub初析 -> Agnes复核",
            "tokenhub",
            args.tokenhub_model,
            tokenhub_api_key or "",
            "agnes",
            args.agnes_model,
            agnes_api_key or "",
            text,
            raw_segments,
            expected,
            args.parse_batch_size,
            args.agnes_parse_max_tokens,
            args.review_batch_size,
            args.agnes_review_max_tokens,
        ))
    if args.only in {"agnes-tokenhub", "tokenhub-both"}:
        if not agnes_api_key:
            raise SystemExit("Please set AGNES_API_KEY in the environment for Agnes parse.")
        chains.append(run_chain(
            "Agnes初析 -> TokenHub复核",
            "agnes",
            args.agnes_model,
            agnes_api_key,
            "tokenhub",
            args.tokenhub_model,
            tokenhub_api_key or "",
            text,
            raw_segments,
            expected,
            args.parse_batch_size,
            args.agnes_parse_max_tokens,
            args.review_batch_size,
            args.agnes_review_max_tokens,
        ))
    if args.only in {"openrouter-agnes", "openrouter-both"}:
        if not agnes_api_key:
            raise SystemExit("Please set AGNES_API_KEY in the environment for Agnes review.")
        chains.append(run_chain(
            "OpenRouter初析 -> Agnes复核",
            "openrouter",
            args.openrouter_model,
            openrouter_api_key or "",
            "agnes",
            args.agnes_model,
            agnes_api_key,
            text,
            raw_segments,
            expected,
            args.parse_batch_size,
            args.agnes_parse_max_tokens,
            args.review_batch_size,
            args.agnes_review_max_tokens,
        ))
    if args.only in {"agnes-openrouter", "openrouter-both"}:
        if not agnes_api_key:
            raise SystemExit("Please set AGNES_API_KEY in the environment for Agnes parse.")
        chains.append(run_chain(
            "Agnes初析 -> OpenRouter复核",
            "agnes",
            args.agnes_model,
            agnes_api_key,
            "openrouter",
            args.openrouter_model,
            openrouter_api_key or "",
            text,
            raw_segments,
            expected,
            args.parse_batch_size,
            args.agnes_parse_max_tokens,
            args.review_batch_size,
            args.agnes_review_max_tokens,
        ))

    output = {
        "source": str(args.source),
        "sample": str(args.sample),
        "text_chars": len(text),
        "total_expected_cases": len(expected),
        "chains": chains,
        "summary": [compact_chain(chain) for chain in chains],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
