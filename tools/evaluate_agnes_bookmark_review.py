from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SAMPLE = Path(r"I:\code\aitts\omnivoice-reader\docs\samples\muli4_part001_first_hour_bookmark_regression.json")
DEFAULT_OUT = Path(r"I:\code\aitts\omnivoice-reader\outputs\temp_by_date\2026-06-06\agnes_bookmark_review_eval.json")
DEFAULT_URL = "https://apihub.agnes-ai.com/v1/chat/completions"
DEFAULT_MODEL = "agnes-2.0-flash"

ROLE_HINTS: dict[str, Any] = {
    "甘织玲奈子": ["玲奈子", "小玲奈", "玲奈亲", "甘织同学", "玲奈无尾熊", "姊姊", "玲奈子小姐", "甘织"],
    "王冢真唯": ["真唯", "小真唯", "王冢同学", "完美恋人", "王冢真唯小姐", "大小姐", "真唯前辈"],
    "濑名紫阳花": ["紫阳花", "紫阳花同学", "小紫", "濑名"],
    "琴纱月": ["纱月", "纱月同学", "小纱", "moon", "moon小姐", "琴纱月小姐", "小纱月"],
    "小柳香穗": ["小香穗", "香穗", "皆口香穗", "nagipo", "nagipo小姐", "小柳同学", "香穗小姐"],
    "甘织遥奈": ["遥奈", "妹妹", "次女"],
    "王冢芮妮": ["芮妮", "妈妈", "母亲", "真唯妈妈"],
    "花取": ["花取小姐"],
    "星来": ["赛菈菈", "赛拉菈·赛菈菈菈菈", "星来学妹", "赛菈菈小姐"],
    "美知留": ["美知留老师"],
    "帕曼小姐": [],
    "米海尔小姐": [],
    "艾玛小姐": [],
    "玲奈子妈妈": {"aliases": ["妈妈", "母亲"], "owner": "甘织玲奈子"},
    "长谷川同学": [],
    "平野同学": [],
    "厕所女生A": [],
    "厕所女生B": [],
    "厕所女生C": [],
    "厕所女生D": [],
    "未知临时人物": [],
}


def strip_generated_labels(text: str) -> str:
    # Surrounding LRC/snapshot text may contain wrong generated labels. Remove short
    # line prefixes so the reviewer judges from prose context instead of trusting them.
    return re.sub(r"(^|\n)[^\n：:]{1,14}[：:]", r"\1", str(text or ""))


def extract_json_array(content: str) -> list[dict[str, Any]]:
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    match = re.search(r"\[.*\]", content, flags=re.S)
    if match:
        content = match.group(0)
    parsed = json.loads(content)
    if isinstance(parsed, dict):
        for key in ("output", "results", "items"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}: {content[:200]}")
    return parsed


def build_case(case: dict[str, Any], raw_segments: list[dict[str, Any]]) -> dict[str, Any]:
    idx = int(case["index"])
    seg = raw_segments[idx]
    candidates = list(ROLE_HINTS.keys())
    return {
        "index": idx,
        "current_speaker_may_be_wrong": seg.get("speaker"),
        "quote": seg.get("text"),
        "context_before": strip_generated_labels(str(seg.get("context_before") or "")[-900:]),
        "context_after": strip_generated_labels(str(seg.get("context_after") or "")[:900]),
        "candidates": [{"label": f"C{i + 1}", "name": name} for i, name in enumerate(candidates)],
    }


def call_agnes(url: str, api_key: str, model: str, cases: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, Any]], str]:
    system = (
        "You are a deterministic Chinese novel speaker-attribution reviewer. "
        "Do not think out loud. Do not explain. Return final JSON only. "
        "For each case choose exactly one candidate label. "
        "Important: a name/nickname inside a quote usually means addressee, not speaker. "
        "First-person narrator is 甘织玲奈子 unless context explicitly switches."
    )
    user = {
        "task": "For each case, identify the true speaker of quote from candidates.",
        "narrator": "甘织玲奈子",
        "role_hints": ROLE_HINTS,
        "cases": cases,
        "output_schema": [
            {
                "index": "number from input",
                "speaker_label": "candidate label such as C1",
                "speaker": "exact candidate name",
                "confidence": "0-1",
                "evidence": "short Chinese reason",
            }
        ],
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"].get("content") or ""
    return extract_json_array(content), data["choices"][0].get("finish_reason") or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Agnes API on bookmark speaker-review regression cases.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2500)
    args = parser.parse_args()

    api_key = os.getenv("AGNES_API_KEY")
    if not api_key:
        raise SystemExit("Please set AGNES_API_KEY in the environment.")

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    raw_segments = json.loads(Path(sample["source"]["raw_snapshot"]).read_text(encoding="utf-8"))["segments"]
    expected = {int(case["index"]): str(case["expected_speaker"]) for case in sample["corrected_segments"]}

    all_results: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    cases = sample["corrected_segments"]
    for start in range(0, len(cases), args.batch_size):
        batch_cases = cases[start : start + args.batch_size]
        payload_cases = [build_case(case, raw_segments) for case in batch_cases]
        try:
            results, finish_reason = call_agnes(args.url, api_key, args.model, payload_cases, args.max_tokens)
        except Exception as exc:
            raw_batches.append({"start": start, "error": repr(exc), "case_indices": [case["index"] for case in batch_cases]})
            continue
        raw_batches.append({"start": start, "finish_reason": finish_reason, "case_indices": [case["index"] for case in batch_cases]})
        all_results.extend(results)
        time.sleep(0.5)

    normalized_results: list[dict[str, Any]] = []
    correct = 0
    for result in all_results:
        idx = int(result.get("index"))
        got = str(result.get("speaker") or "").strip()
        exp = expected.get(idx, "")
        ok = got == exp
        correct += int(ok)
        normalized_results.append({
            "index": idx,
            "expected": exp,
            "speaker": got,
            "ok": ok,
            "confidence": result.get("confidence"),
            "evidence": result.get("evidence"),
            "raw": result,
        })

    output = {
        "model": args.model,
        "total_expected": len(expected),
        "total_returned": len(normalized_results),
        "correct": correct,
        "accuracy": correct / max(1, len(normalized_results)),
        "batches": raw_batches,
        "results": sorted(normalized_results, key=lambda item: item["index"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if len(normalized_results) != len(expected):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
