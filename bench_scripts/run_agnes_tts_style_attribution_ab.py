from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "docs" / "samples"
OUT_ROOT = REPO / "bench_outputs" / "tts_style_attribution"
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_MODEL = "agnes-2.0-flash"


def load_dotenv_if_needed() -> None:
    if os.environ.get("AGNES_API_KEY"):
        return
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_name(value: object) -> str:
    text = str(value or "").strip().replace("瀬", "濑")
    aliases = {
        "叙述者": "旁白",
        "我": "甘织玲奈子",
        "玲奈子": "甘织玲奈子",
        "香穗": "小柳香穗",
        "小香穗": "小柳香穗",
        "纱月": "琴纱月",
        "紫阳花": "濑名紫阳花",
        "小紫": "濑名紫阳花",
        "真唯": "王冢真唯",
    }
    return aliases.get(text, text)


def score_against_groundtruth(parsed_segments: list[dict[str, Any]], groundtruth_path: Path) -> dict[str, Any]:
    gt = json.loads(groundtruth_path.read_text(encoding="utf-8"))
    gt_segments = gt.get("segments") or []
    total = correct = missing = crowd = 0
    wrong: list[dict[str, Any]] = []
    for item in gt_segments:
        idx = int(item.get("i", -1))
        if item.get("crowd"):
            crowd += 1
            continue
        expected = normalize_name(item.get("speaker"))
        if not expected:
            continue
        total += 1
        if idx < 0 or idx >= len(parsed_segments):
            missing += 1
            wrong.append({"i": idx, "expected": expected, "got": None, "text": item.get("text")})
            continue
        got = normalize_name(parsed_segments[idx].get("speaker"))
        ok = got == expected
        correct += int(ok)
        if not ok:
            wrong.append({
                "i": idx,
                "expected": expected,
                "got": got,
                "text": item.get("text"),
                "confidence": parsed_segments[idx].get("confidence"),
                "evidence": parsed_segments[idx].get("evidence"),
            })
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "missing": missing,
        "crowd_excluded": crowd,
        "wrong_count": len(wrong),
        "wrong": wrong,
    }


def summarize_tts_style_raw(raw_outputs: list[str]) -> dict[str, Any]:
    rows = 0
    rows_with_style = 0
    examples: list[dict[str, str]] = []
    for output in raw_outputs:
        for line in str(output or "").splitlines():
            parts = [part.strip() for part in line.strip().split("|")]
            if len(parts) < 5 or not parts[0].strip(" .、").isdigit():
                continue
            rows += 1
            if len(parts) >= 6 and parts[5]:
                rows_with_style += 1
                if len(examples) < 8:
                    examples.append({
                        "row": parts[0],
                        "speaker": parts[1] if len(parts) > 1 else "",
                        "evidence": parts[4] if len(parts) > 4 else "",
                        "tts_style": parts[5],
                    })
    return {
        "raw_outputs": len(raw_outputs),
        "compact_rows": rows,
        "rows_with_tts_style": rows_with_style,
        "style_coverage": round(rows_with_style / rows, 4) if rows else 0.0,
        "examples": examples,
    }


def install_tts_style_prompt_patch(raw_outputs: list[str]) -> None:
    import book_voice_parser.batch_llm_attributor as bla

    style_rules = """

【额外任务：同时给 IndexTTS2 生成语气描述】
在判断 speaker 的同时，为该句台词生成一个给 TTS 使用的短语气描述。它必须服务于朗读，不参与改写文本。
格式建议：语气/情绪 + 语速/音量 + 表演约束，例如「语气轻快带笑，语速中等，音量自然」。
要求：
- 12-28 个汉字，避免角色名、避免解释剧情，避免英文标签。
- 只能根据本句和紧邻上下文判断；不确定时写「自然平稳，语速中等」。
- 不要为了匹配语气而改变 speaker 判断；speaker 仍按上面的证据优先级裁决。
"""
    bla._SYSTEM_PROMPT_COMPACT = bla._SYSTEM_PROMPT_COMPACT.replace(
        "【输出格式】",
        style_rules + "\n【输出格式】",
    ).replace(
        "每行一条，格式：序号|说话人|置信度|类型|依据(≤15字)",
        "每行一条，格式：序号|说话人|置信度|类型|依据(≤15字)|tts_style(12-28字)",
    ).replace(
        "1|王冢真唯|0.92|eb|前文「真唯说道」",
        "1|王冢真唯|0.92|eb|前文「真唯说道」|语气从容认真，语速中等",
    ).replace(
        "2|甘织玲奈子|0.70|im|上句真唯，轮换",
        "2|甘织玲奈子|0.70|im|上句真唯，轮换|语气迟疑小声，略带紧张",
    )
    bla._USER_TEMPLATE_COMPACT = bla._USER_TEMPLATE_COMPACT.replace(
        "请对以下 {count} 条台词逐一归因，每行输出：序号|说话人|置信度|类型|依据",
        "请对以下 {count} 条台词逐一归因，并同时给 IndexTTS2 生成语气描述。每行输出：序号|说话人|置信度|类型|依据|tts_style",
    )

    original_call_llm = bla.BatchLLMAttributor._call_llm

    def wrapped_call_llm(self, system: str, user: str, max_tokens: int | None = None) -> str:
        text = original_call_llm(self, system, user, max_tokens=max_tokens)
        raw_outputs.append(text)
        return text

    bla.BatchLLMAttributor._call_llm = wrapped_call_llm


def run_parse(seg: str, *, variant: str, out_dir: Path, batch_size: int, max_tokens: int, timeout: int) -> dict[str, Any]:
    sys.path.insert(0, str(REPO / "BookVoiceParser"))
    sys.path.insert(0, str(REPO / "tools"))
    from book_voice_parser import BatchConfig, parse_novel
    from evaluate_agnes_bookmark_review import ROLE_HINTS

    raw_outputs: list[str] = []
    if variant == "tts_style":
        install_tts_style_prompt_patch(raw_outputs)

    text = (SAMPLES / f"{seg}_sample.txt").read_text(encoding="utf-8")
    cfg = BatchConfig(
        base_url=AGNES_BASE_URL,
        api_key=os.environ["AGNES_API_KEY"],
        model=AGNES_MODEL,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=0.0,
        timeout=timeout,
        context_chars=320,
        output_mode="compact",
        disable_thinking=True,
    )
    started = time.perf_counter()
    result = parse_novel(
        text,
        role_hints=ROLE_HINTS,
        batch_llm_config=cfg,
        narrator="甘织玲奈子",
        return_result=True,
        include_narration=False,
        review_threshold=0.7,
        enable_block_review=True,
    )
    elapsed = round(time.perf_counter() - started, 2)
    segments = [seg.model_dump(mode="json") for seg in result.segments]
    score = score_against_groundtruth(segments, SAMPLES / f"{seg}_groundtruth.json")
    style_summary = summarize_tts_style_raw(raw_outputs) if variant == "tts_style" else None
    output = {
        "variant": variant,
        "model": AGNES_MODEL,
        "sample": seg,
        "text_chars": len(text),
        "elapsed_sec": elapsed,
        "parsed_segments": len(segments),
        "score": score,
        "style_summary": style_summary,
        "stats": result.stats,
        "segments": segments,
    }
    parse_path = out_dir / f"{seg}_{variant}_parse.json"
    parse_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test whether adding IndexTTS-style tone descriptions affects speaker attribution.")
    parser.add_argument("--seg", default="muli4_seg8")
    parser.add_argument("--variant", choices=["baseline", "tts_style", "both"], default="tts_style")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    load_dotenv_if_needed()
    if not os.environ.get("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY is not set.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["baseline", "tts_style"] if args.variant == "both" else [args.variant]
    results = []
    for variant in variants:
        print(f"[run] {args.seg} / {variant} / Agnes mature pipeline", flush=True)
        results.append(run_parse(
            args.seg,
            variant=variant,
            out_dir=out_dir,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        ))
        score = results[-1]["score"]
        print(f"[score] {variant}: {score['correct']}/{score['total']} = {score['accuracy']:.2%}", flush=True)

    existing_parse = json.loads((SAMPLES / f"{args.seg}_parse.json").read_text(encoding="utf-8"))
    existing_segments = existing_parse.get("segments") or []
    existing_score = score_against_groundtruth(existing_segments, SAMPLES / f"{args.seg}_groundtruth.json")
    summary = {
        "sample": args.seg,
        "model": AGNES_MODEL,
        "existing_reference_parse_score": existing_score,
        "runs": [
            {
                "variant": item["variant"],
                "elapsed_sec": item["elapsed_sec"],
                "parsed_segments": item["parsed_segments"],
                "score": {k: v for k, v in item["score"].items() if k != "wrong"},
                "style_summary": item.get("style_summary"),
                "wrong_ids": [w["i"] for w in item["score"]["wrong"]],
            }
            for item in results
        ],
    }
    if len(results) == 2:
        base = results[0]["score"]
        style = results[1]["score"]
        summary["delta"] = {
            "accuracy_points": round((style["accuracy"] - base["accuracy"]) * 100, 2),
            "correct_delta": style["correct"] - base["correct"],
            "baseline_wrong_style_fixed": sorted(set(w["i"] for w in base["wrong"]) - set(w["i"] for w in style["wrong"])),
            "style_wrong_baseline_correct": sorted(set(w["i"] for w in style["wrong"]) - set(w["i"] for w in base["wrong"])),
        }
    summary_path = out_dir / f"{args.seg}_tts_style_ab_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[out] {summary_path}")


if __name__ == "__main__":
    main()
