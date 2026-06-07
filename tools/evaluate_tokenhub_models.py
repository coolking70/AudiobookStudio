from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from evaluate_agnes_bookmark_review import ROLE_HINTS, build_case, extract_json_array


DEFAULT_SAMPLE = Path(r"I:\code\aitts\omnivoice-reader\docs\samples\muli4_part001_first_hour_bookmark_regression.json")
DEFAULT_OUT = Path(r"I:\code\aitts\omnivoice-reader\outputs\temp_by_date\2026-06-07\tokenhub_models_review_eval.json")
DEFAULT_URL = "https://tokenhub.tencentmaas.com/v1/chat/completions"
DEFAULT_MODELS = [
    "deepseek-r1-0528",
    "glm-5-turbo",
    "qwen3.5-plus",
    "hunyuan-role-latest",
    "glm-5",
    "minimax-m3",
    "glm-5.1",
    "deepseek-v3.1-terminus",
    "hy-mt2-pro",
    "kimi-k2.5",
    "hunyuan-2.0-thinking-20251109",
    "minimax-m2.7",
    "qwen3.5-flash",
    "kimi-k2.6",
    "glm-5v-turbo",
    "hunyuan-2.0-instruct-20251111",
    "minimax-m2.5",
]


def _post_chat(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    *,
    retry_without_enable_thinking: bool = True,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        if retry_without_enable_thinking and "enable_thinking" in payload and exc.code in {400, 422}:
            next_payload = dict(payload)
            next_payload.pop("enable_thinking", None)
            return _post_chat(url, api_key, next_payload, timeout, retry_without_enable_thinking=False)
        raise RuntimeError(f"HTTP {exc.code}: {body[:1200]}") from exc


def _message_content(data: dict[str, Any]) -> tuple[str, str, str]:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    reasoning = msg.get("reasoning_content") or ""
    return str(content or "").strip(), str(reasoning or ""), str(choice.get("finish_reason") or "")


def connectivity_test(url: str, api_key: str, model: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return final text only. No reasoning. Output exactly OK."},
            {"role": "user", "content": "请只输出 OK"},
        ],
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
        "enable_thinking": False,
    }
    try:
        data = _post_chat(url, api_key, payload, timeout)
        content, reasoning, finish = _message_content(data)
        return {
            "ok": bool(content),
            "seconds": time.time() - started,
            "content": content[:300],
            "reasoning_len": len(reasoning),
            "finish_reason": finish,
        }
    except Exception as exc:
        return {"ok": False, "seconds": time.time() - started, "error": repr(exc)}


def build_review_payload(model: str, cases: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
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
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
    }


def evaluate_review_model(
    url: str,
    api_key: str,
    model: str,
    sample: dict[str, Any],
    raw_segments: list[dict[str, Any]],
    expected: dict[int, str],
    batch_size: int,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    started = time.time()
    connectivity = connectivity_test(url, api_key, model, timeout)
    all_results: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    corrected_cases = list(sample["corrected_segments"])
    for start in range(0, len(corrected_cases), batch_size):
        batch_cases = corrected_cases[start : start + batch_size]
        payload_cases = [build_case(case, raw_segments) for case in batch_cases]
        batch_record: dict[str, Any] = {
            "start": start,
            "case_indices": [int(case["index"]) for case in batch_cases],
        }
        batch_started = time.time()
        try:
            data = _post_chat(
                url,
                api_key,
                build_review_payload(model, payload_cases, max_tokens),
                timeout,
            )
            content, reasoning, finish = _message_content(data)
            batch_record.update({
                "seconds": time.time() - batch_started,
                "finish_reason": finish,
                "content_len": len(content),
                "reasoning_len": len(reasoning),
            })
            if not content:
                raise RuntimeError(f"empty content; reasoning_len={len(reasoning)}; finish={finish}")
            parsed = extract_json_array(content)
            all_results.extend(parsed)
        except Exception as exc:
            batch_record.update({
                "seconds": time.time() - batch_started,
                "error": repr(exc),
            })
        batches.append(batch_record)

    normalized_results: list[dict[str, Any]] = []
    correct = 0
    for result in all_results:
        try:
            idx = int(result.get("index"))
        except Exception:
            continue
        got = str(result.get("speaker") or "").strip()
        got = re.sub(r"^角色[:：]\s*", "", got)
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

    returned = len(normalized_results)
    return {
        "model": model,
        "connectivity": connectivity,
        "total_expected": len(expected),
        "total_returned": returned,
        "correct": correct,
        "accuracy_returned": correct / max(1, returned),
        "accuracy_total": correct / max(1, len(expected)),
        "seconds": time.time() - started,
        "batches": batches,
        "results": sorted(normalized_results, key=lambda item: item["index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple TokenHub models on speaker-attribution review cases.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    api_key = os.getenv("TOKENHUB_API_KEY")
    if not api_key:
        raise SystemExit("Please set TOKENHUB_API_KEY in the environment.")

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    raw_segments = json.loads(Path(sample["source"]["raw_snapshot"]).read_text(encoding="utf-8"))["segments"]
    expected = {int(case["index"]): str(case["expected_speaker"]) for case in sample["corrected_segments"]}

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                evaluate_review_model,
                args.url,
                api_key,
                model,
                sample,
                raw_segments,
                expected,
                args.batch_size,
                args.max_tokens,
                args.timeout,
            ): model
            for model in args.models
        }
        for future in concurrent.futures.as_completed(futures):
            model = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"model": model, "error": repr(exc)}
            results.append(result)
            returned = result.get("total_returned", 0)
            correct = result.get("correct", 0)
            print(
                f"{model}: returned={returned}, correct={correct}, "
                f"acc_total={result.get('accuracy_total', 0):.3f}, "
                f"seconds={result.get('seconds', 0):.1f}",
                flush=True,
            )

    results.sort(key=lambda item: (item.get("accuracy_total", 0), item.get("total_returned", 0)), reverse=True)
    output = {
        "sample": str(args.sample),
        "models": args.models,
        "summary": [
            {
                "model": item.get("model"),
                "connectivity_ok": bool((item.get("connectivity") or {}).get("ok")),
                "total_returned": item.get("total_returned", 0),
                "correct": item.get("correct", 0),
                "accuracy_total": item.get("accuracy_total", 0),
                "accuracy_returned": item.get("accuracy_returned", 0),
                "seconds": item.get("seconds", 0),
                "batch_errors": sum(1 for batch in item.get("batches", []) if batch.get("error")),
            }
            for item in results
        ],
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
