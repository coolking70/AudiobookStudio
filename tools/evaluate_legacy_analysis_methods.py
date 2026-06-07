from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluate_agnes_batch_parse_sample import extract_text, normalize_text_for_match
from evaluate_agnes_bookmark_review import ROLE_HINTS


DEFAULT_SOURCE = Path(r"C:\Users\coolking\Downloads\125697 utf-8.txt")
DEFAULT_SAMPLE = Path(r"I:\code\aitts\omnivoice-reader\docs\samples\muli4_part001_first_hour_bookmark_regression.json")
DEFAULT_OUT = Path(r"I:\code\aitts\omnivoice-reader\outputs\temp_by_date\2026-06-07\legacy_analysis_methods_agnes_eval.json")


@dataclass
class MethodResult:
    method: str
    seconds: float
    segments: int
    exact_correct: int
    exact_matched: int
    exact_total_accuracy: float
    exact_matched_accuracy: float
    fuzzy_correct: int
    fuzzy_matched: int
    fuzzy_total_accuracy: float
    fuzzy_matched_accuracy: float
    error: str | None = None
    stats: dict[str, Any] | None = None
    cases: list[dict[str, Any]] | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_llm_config(model: str, api_key: str, *, context_mode: str = "chapter", workers: int = 1):
    sys.path.insert(0, str(repo_root()))
    from schemas import LLMConfig
    from role_analyzer import (
        CHARACTER_ALIAS_RESOLUTION_PROMPT,
        DEFAULT_SYSTEM_PROMPT,
        SEGMENT_PLAN_PROMPT,
        SPEAKER_VERIFICATION_PROMPT,
        TEXT_OPTIMIZATION_PROMPT,
    )

    return LLMConfig(
        base_url="https://apihub.agnes-ai.com/v1/chat/completions",
        api_key=api_key,
        model=model,
        max_tokens=5000,
        temperature=0.0,
        timeout=240,
        workers=workers,
        context_mode=context_mode,
        output_mode="compact",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        segment_prompt=SEGMENT_PLAN_PROMPT,
        text_optimization_prompt=TEXT_OPTIMIZATION_PROMPT,
        speaker_verification_prompt=SPEAKER_VERIFICATION_PROMPT,
        alias_resolution_prompt=CHARACTER_ALIAS_RESOLUTION_PROMPT,
        enable_character_alias_merge=False,
        enable_speaker_verification=False,
    )


def load_expected(sample_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, str]]:
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    raw_segments = json.loads(Path(sample["source"]["raw_snapshot"]).read_text(encoding="utf-8"))["segments"]
    expected = {int(case["index"]): str(case["expected_speaker"]) for case in sample["corrected_segments"]}
    return sample, raw_segments, expected


def segment_text(seg: Any) -> str:
    if isinstance(seg, dict):
        return str(seg.get("text") or "")
    return str(getattr(seg, "text", "") or "")


def segment_speaker(seg: Any) -> str:
    if isinstance(seg, dict):
        return str(seg.get("speaker") or "")
    return str(getattr(seg, "speaker", "") or "")


def segment_confidence(seg: Any) -> Any:
    if isinstance(seg, dict):
        return seg.get("confidence")
    return getattr(seg, "confidence", None)


def segment_evidence(seg: Any) -> Any:
    if isinstance(seg, dict):
        return seg.get("evidence") or seg.get("_evidence")
    return getattr(seg, "evidence", None)


def evaluate(segments: list[Any], raw_segments: list[dict[str, Any]], expected: dict[int, str]) -> dict[str, Any]:
    parsed_by_text: dict[str, list[tuple[int, Any]]] = {}
    normalized_texts: list[str] = []
    for idx, seg in enumerate(segments):
        key = normalize_text_for_match(segment_text(seg))
        normalized_texts.append(key)
        parsed_by_text.setdefault(key, []).append((idx, seg))

    cases: list[dict[str, Any]] = []
    exact_correct = 0
    exact_matched = 0
    fuzzy_correct = 0
    fuzzy_matched = 0

    for source_index, expected_speaker in expected.items():
        raw_text = str(raw_segments[source_index].get("text") or "")
        key = normalize_text_for_match(raw_text)
        exact_matches = parsed_by_text.get(key) or []
        exact_case: dict[str, Any] = {
            "source_index": source_index,
            "expected": expected_speaker,
            "text": raw_text,
        }
        if exact_matches:
            exact_matched += 1
            parsed_index, seg = exact_matches[0]
            got = segment_speaker(seg)
            ok = got == expected_speaker
            exact_correct += int(ok)
            exact_case.update({
                "exact_index": parsed_index,
                "exact_speaker": got,
                "exact_ok": ok,
                "confidence": segment_confidence(seg),
                "evidence": segment_evidence(seg),
            })
        else:
            exact_case.update({"exact_missing": True, "exact_ok": False})

        fuzzy_idx = None
        for idx, text_key in enumerate(normalized_texts):
            if not key:
                continue
            if key in text_key or text_key in key:
                fuzzy_idx = idx
                break
        if fuzzy_idx is not None:
            fuzzy_matched += 1
            seg = segments[fuzzy_idx]
            got = segment_speaker(seg)
            ok = got == expected_speaker
            fuzzy_correct += int(ok)
            exact_case.update({
                "fuzzy_index": fuzzy_idx,
                "fuzzy_speaker": got,
                "fuzzy_ok": ok,
            })
        else:
            exact_case.update({"fuzzy_missing": True, "fuzzy_ok": False})
        cases.append(exact_case)

    total = len(expected)
    return {
        "exact_correct": exact_correct,
        "exact_matched": exact_matched,
        "exact_total_accuracy": exact_correct / max(1, total),
        "exact_matched_accuracy": exact_correct / max(1, exact_matched),
        "fuzzy_correct": fuzzy_correct,
        "fuzzy_matched": fuzzy_matched,
        "fuzzy_total_accuracy": fuzzy_correct / max(1, total),
        "fuzzy_matched_accuracy": fuzzy_correct / max(1, fuzzy_matched),
        "cases": cases,
    }


def run_bvp_method(name: str, text: str, raw_segments: list[dict[str, Any]], expected: dict[int, str], llm_config: Any | None, implicit_strategy: str) -> MethodResult:
    sys.path.insert(0, str(repo_root() / "BookVoiceParser"))
    from book_voice_parser import parse_novel

    started = time.time()
    result = parse_novel(
        text,
        role_hints=ROLE_HINTS,
        llm_config=llm_config,
        batch_llm_config=None,
        narrator="甘织玲奈子",
        return_result=True,
        include_narration=False,
        review_threshold=0.7,
        implicit_strategy=implicit_strategy,
    )
    metrics = evaluate(result.segments, raw_segments, expected)
    return MethodResult(
        method=name,
        seconds=time.time() - started,
        segments=len(result.segments),
        stats=result.stats,
        **metrics,
    )


def run_role_analyzer_method(name: str, text: str, raw_segments: list[dict[str, Any]], expected: dict[int, str], llm_config: Any, combo: str) -> MethodResult:
    sys.path.insert(0, str(repo_root()))
    from role_analyzer import (
        analyze_chunks_with_llm,
        intelligent_segment_text,
        optimize_and_analyze_chunks_with_llm,
        optimize_chunks_for_tts,
        segment_and_optimize_text_with_info,
        verify_speakers_pass,
    )

    started = time.time()
    if combo == "segment_then_analyze":
        chunks = intelligent_segment_text(text, llm_config)
        segments = analyze_chunks_with_llm(chunks, llm_config)
    elif combo == "segment_optimize_then_analyze":
        optimized_chunks, _info = segment_and_optimize_text_with_info(text, llm_config)
        segments = analyze_chunks_with_llm(optimized_chunks, llm_config)
    elif combo == "segment_then_optimize_analyze":
        chunks = intelligent_segment_text(text, llm_config)
        segments = optimize_and_analyze_chunks_with_llm(chunks, llm_config)
    elif combo == "separate":
        chunks = intelligent_segment_text(text, llm_config)
        optimized_chunks = optimize_chunks_for_tts(chunks, llm_config)
        segments = analyze_chunks_with_llm(optimized_chunks, llm_config)
    elif combo == "segment_then_analyze_verify":
        chunks = intelligent_segment_text(text, llm_config)
        segments = analyze_chunks_with_llm(chunks, llm_config)
        segments = verify_speakers_pass(segments, llm_config)
    else:
        raise ValueError(f"unknown combo: {combo}")

    metrics = evaluate(segments, raw_segments, expected)
    return MethodResult(
        method=name,
        seconds=time.time() - started,
        segments=len(segments),
        **metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-BatchLLM legacy analysis methods with Agnes.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="agnes-2.0-flash")
    parser.add_argument("--methods", nargs="*", default=[
        "bvp_heuristic",
        "bvp_llm_spc",
        "legacy_segment_then_analyze",
        "legacy_segment_then_analyze_verify",
        "legacy_segment_then_optimize_analyze",
    ])
    args = parser.parse_args()

    api_key = os.getenv("AGNES_API_KEY")
    if not api_key:
        raise SystemExit("Please set AGNES_API_KEY in the environment.")

    text = extract_text(args.source)
    _sample, raw_segments, expected = load_expected(args.sample)
    llm = make_llm_config(args.model, api_key, context_mode="chapter", workers=1)

    results: list[MethodResult] = []
    for method in args.methods:
        print(f"RUN {method}", flush=True)
        try:
            if method == "bvp_heuristic":
                result = run_bvp_method("BVP classic heuristic", text, raw_segments, expected, None, "heuristic")
            elif method == "bvp_llm_spc":
                result = run_bvp_method("BVP classic llm_spc", text, raw_segments, expected, llm, "llm_spc")
            elif method.startswith("legacy_"):
                combo = method.removeprefix("legacy_")
                result = run_role_analyzer_method(f"legacy {combo}", text, raw_segments, expected, llm, combo)
            else:
                raise ValueError(f"unknown method: {method}")
        except Exception as exc:
            result = MethodResult(
                method=method,
                seconds=0,
                segments=0,
                exact_correct=0,
                exact_matched=0,
                exact_total_accuracy=0,
                exact_matched_accuracy=0,
                fuzzy_correct=0,
                fuzzy_matched=0,
                fuzzy_total_accuracy=0,
                fuzzy_matched_accuracy=0,
                error=repr(exc),
            )
        results.append(result)
        print(
            f"{result.method}: exact {result.exact_correct}/30 "
            f"fuzzy {result.fuzzy_correct}/30 segments={result.segments} "
            f"seconds={result.seconds:.1f} error={result.error or ''}",
            flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "source": str(args.source),
            "sample": str(args.sample),
            "model": args.model,
            "text_chars": len(text),
            "results": [asdict(item) for item in results],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = [
        {
            "method": item.method,
            "segments": item.segments,
            "exact": f"{item.exact_correct}/30",
            "exact_accuracy": item.exact_total_accuracy,
            "exact_matched": item.exact_matched,
            "fuzzy": f"{item.fuzzy_correct}/30",
            "fuzzy_accuracy": item.fuzzy_total_accuracy,
            "fuzzy_matched": item.fuzzy_matched,
            "seconds": item.seconds,
            "error": item.error,
        }
        for item in results
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
