from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evaluate_agnes_bookmark_review import ROLE_HINTS


DEFAULT_SOURCE = Path(r"C:\Users\coolking\Downloads\125697 utf-8.txt")
DEFAULT_SAMPLE = Path(r"I:\code\aitts\omnivoice-reader\docs\samples\muli4_part001_first_hour_bookmark_regression.json")
DEFAULT_OUT = Path(r"I:\code\aitts\omnivoice-reader\outputs\temp_by_date\2026-06-06\agnes_batch_parse_sample_eval.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def extract_text(source: Path) -> str:
    try:
        text = source.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = source.read_text(encoding="utf-16")
    start = text.find("第四卷 序章")
    if start < 0:
        start = 0
    marker = "是为了欢迎我回来"
    end = text.find(marker, start)
    if end < 0:
        end = min(len(text), start + 70000)
    else:
        end += len(marker) + 80
    return text[start:end].strip()


def normalize_text_for_match(value: str) -> str:
    # collapse whitespace + reconcile traditional/simplified variants found in the
    # Taiwan-edition original ("彷佛"/"仿佛", "姊姊"/"姐姐").
    return "".join(str(value or "").split()).replace("彷佛", "仿佛").replace("姊", "姐")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agnes BatchLLM parse on the bookmark sample text and evaluate cases.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=5000)
    args = parser.parse_args()

    api_key = os.getenv("AGNES_API_KEY")
    if not api_key:
        raise SystemExit("Please set AGNES_API_KEY in the environment.")

    sys.path.insert(0, str(repo_root() / "BookVoiceParser"))
    from book_voice_parser import BatchConfig, parse_novel

    text = extract_text(args.source)
    cfg = BatchConfig(
        base_url="https://apihub.agnes-ai.com/v1",
        api_key=api_key,
        model="agnes-2.0-flash",
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=0.0,
        timeout=180,
        context_chars=320,
        output_mode="compact",
        disable_thinking=True,
    )
    result = parse_novel(
        text,
        role_hints=ROLE_HINTS,
        batch_llm_config=cfg,
        narrator="甘织玲奈子",
        return_result=True,
        include_narration=False,
        review_threshold=0.7,
    )
    parsed_segments = [seg.model_dump(mode="json") for seg in result.segments]
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    raw_snapshot = json.loads(Path(sample["source"]["raw_snapshot"]).read_text(encoding="utf-8"))["segments"]

    parsed_by_text: dict[str, list[dict]] = {}
    for seg in parsed_segments:
        parsed_by_text.setdefault(normalize_text_for_match(seg.get("text", "")), []).append(seg)

    cases = []
    correct = 0
    for case in sample["corrected_segments"]:
        idx = int(case["index"])
        expected = str(case["expected_speaker"])
        raw_text = str(raw_snapshot[idx].get("text") or "")
        key = normalize_text_for_match(raw_text)
        matches = parsed_by_text.get(key) or []
        if matches:
            got = str(matches[0].get("speaker") or "")
            ok = got == expected
            correct += int(ok)
            cases.append({
                "index": idx,
                "expected": expected,
                "speaker": got,
                "ok": ok,
                "text": raw_text,
                "confidence": matches[0].get("confidence"),
                "evidence": matches[0].get("evidence"),
                "quote_id": matches[0].get("quote_id"),
            })
        else:
            cases.append({
                "index": idx,
                "expected": expected,
                "speaker": None,
                "ok": False,
                "text": raw_text,
                "missing": True,
            })

    output = {
        "model": "agnes-2.0-flash",
        "source": str(args.source),
        "text_chars": len(text),
        "parsed_segments": len(parsed_segments),
        "total_cases": len(cases),
        "matched_cases": sum(1 for item in cases if not item.get("missing")),
        "correct": correct,
        "accuracy": correct / max(1, len(cases)),
        "stats": result.stats,
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
