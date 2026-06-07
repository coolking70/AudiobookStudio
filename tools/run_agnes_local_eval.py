"""Faithful Agnes BatchLLM analysis on the real original novel slice, with
sub-segment-merge alignment.

Pipeline (project's own `parse_novel` + Agnes BatchLLM backend) runs on the first-hour
slice of the original novel; results are scored against the reviewed snapshot (full
ground truth) and the 30 hard cases.

Why merge-alignment: the reviewed snapshot splits long quotes into several TTS-sized
sub-segments (346 dialogue pieces), while parse_novel emits whole quotes (~224). A naive
1:1 text match leaves ~125 sub-segments "unmatched". Here we walk both ordered sequences
and merge consecutive snapshot dialogue sub-segments to reconstruct each parser quote, so
every parser segment is scored.

The raw parse is cached to docs/samples/_agnes_parse_cache.json so alignment can be
re-tuned without re-spending API quota (delete the cache or pass --refresh to re-parse).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
ORIGINAL = SAMP / "muli4_original_125697_utf8.txt"
SNAPSHOT = SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json"
REGRESSION = SAMP / "muli4_part001_first_hour_bookmark_regression.json"
CACHE = SAMP / "_agnes_parse_cache.json"
OUT = SAMP / "agnes_original_eval_result.json"

sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "BookVoiceParser"))

from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402

START_MARKER = "第四卷 序章"
END_MARKER = "是为了欢迎我回来"


def norm(value: str) -> str:
    return "".join(str(value or "").split()).replace("彷佛", "仿佛").replace("姊", "姐")


def extract_slice() -> str:
    try:
        text = ORIGINAL.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = ORIGINAL.read_text(encoding="utf-16")
    start = text.find(START_MARKER)
    if start < 0:
        start = 0
    end = text.find(END_MARKER, start)
    end = end + len(END_MARKER) + 80 if end >= 0 else min(len(text), start + 70000)
    return text[start:end].strip()


def get_parsed(text: str, refresh: bool) -> tuple[list[dict], dict]:
    if CACHE.exists() and not refresh:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if cached.get("slice_chars") == len(text):
            print(f"[cache] reusing parsed segments from {CACHE.name}", flush=True)
            return cached["segments"], cached.get("stats", {})
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("No cache available and AGNES_API_KEY not set.")
    from book_voice_parser import BatchConfig, parse_novel
    cfg = BatchConfig(
        base_url="https://apihub.agnes-ai.com/v1",
        api_key=os.environ["AGNES_API_KEY"],
        model="agnes-2.0-flash",
        batch_size=8, max_tokens=5000, temperature=0.0, timeout=180,
        context_chars=320, output_mode="compact", disable_thinking=True,
    )
    print("[run] parsing original slice with agnes-2.0-flash ...", flush=True)
    result = parse_novel(
        text, role_hints=ROLE_HINTS, batch_llm_config=cfg, narrator="甘织玲奈子",
        return_result=True, include_narration=False, review_threshold=0.7,
    )
    segments = [seg.model_dump(mode="json") for seg in result.segments]
    CACHE.write_text(json.dumps({"slice_chars": len(text), "segments": segments, "stats": result.stats},
                                ensure_ascii=False), encoding="utf-8")
    return segments, result.stats


def merge_align(gt: list[dict], parsed: list[dict]) -> list[dict]:
    """Two-pointer alignment: reconstruct each parsed quote from consecutive snapshot
    dialogue sub-segments. Returns one row per parsed segment that aligns to >=1 gt piece."""
    i = 0
    rows = []
    for seg in parsed:
        ptext = norm(seg.get("text", ""))
        if not ptext:
            continue
        acc = ""
        covered = []
        j = i
        while j < len(gt) and len(acc) < len(ptext):
            acc += gt[j]["norm"]
            covered.append(gt[j])
            j += 1
        if acc == ptext and covered:
            # All covered sub-segments should share one speaker; use majority.
            spk = Counter(c["speaker"] for c in covered).most_common(1)[0][0]
            rows.append({
                "gt": spk, "text": "".join(c["text"] for c in covered),
                "pred": seg.get("speaker"), "conf": seg.get("confidence"),
                "n_sub": len(covered), "matched": True,
            })
            i = j
        else:
            # No clean reconstruction at this position; try a direct single-piece match
            # without advancing the gt pointer (handles minor ordering slips).
            direct = next((g for g in gt[i:i + 3] if g["norm"] == ptext), None)
            if direct:
                rows.append({"gt": direct["speaker"], "text": direct["text"],
                             "pred": seg.get("speaker"), "conf": seg.get("confidence"),
                             "n_sub": 1, "matched": True})
                # advance past it
                while i < len(gt) and gt[i] is not direct:
                    i += 1
                i += 1
            else:
                rows.append({"gt": None, "text": seg.get("text", ""),
                             "pred": seg.get("speaker"), "matched": False})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore cache and re-parse via API")
    args = ap.parse_args()

    text = extract_slice()
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["segments"]
    reg = json.loads(REGRESSION.read_text(encoding="utf-8"))
    nslice = norm(text)

    gt = []
    for seg in snap:
        sp = str(seg.get("speaker") or "")
        t = seg.get("text") or ""
        if sp and sp != "旁白" and norm(t) and norm(t) in nslice:
            gt.append({"speaker": sp, "norm": norm(t), "text": t})

    parsed, stats = get_parsed(text, args.refresh)
    print(f"[run] slice={len(text)} gt_dialogue={len(gt)} parsed={len(parsed)}", flush=True)

    rows = merge_align(gt, parsed)
    matched = [r for r in rows if r["matched"]]
    crowd = lambda s: (s or "").startswith("厕所女生") or s in {"未知", "未知临时人物"}
    named = [r for r in matched if not crowd(r["gt"])]
    overall_c = sum(1 for r in matched if r["pred"] == r["gt"])
    named_c = sum(1 for r in named if r["pred"] == r["gt"])

    # 30 hard cases: snapshot[index].text -> expected; match a parser quote that CONTAINS it.
    hard = []
    for c in reg["corrected_segments"]:
        idx = int(c["index"])
        nt = norm(snap[idx].get("text", ""))
        pred = None
        for seg in parsed:
            if nt and nt in norm(seg.get("text", "")):
                pred = seg.get("speaker"); break
        exp = str(c["expected_speaker"])
        hard.append({"index": idx, "expected": exp, "pred": pred, "ok": pred == exp,
                     "text": snap[idx].get("text", "")})
    hard_c = sum(1 for h in hard if h["ok"])

    conf = Counter()
    for r in named:
        if r["pred"] != r["gt"]:
            conf[f"{r['gt']} -> {r['pred']}"] += 1

    out = {
        "model": "agnes-2.0-flash", "source": "original novel slice (merge-aligned)",
        "slice_chars": len(text), "gt_dialogue": len(gt), "parsed_segments": len(parsed),
        "aligned": len(matched), "unaligned_parsed": len(rows) - len(matched),
        "overall_accuracy_incl_crowd": round(overall_c / max(1, len(matched)), 4),
        "named_accuracy": round(named_c / max(1, len(named)), 4),
        "named_total": len(named), "named_correct": named_c,
        "hard_cases_accuracy": round(hard_c / max(1, len(hard)), 4),
        "hard_cases_correct": hard_c, "hard_cases_total": len(hard),
        "stats": stats, "confusions": dict(conf.most_common()),
        "rows": rows, "hard_cases": hard,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Agnes 原文切片评测（merge-aligned）===")
    print(f"切片字数            : {len(text)}")
    print(f"真值子片段(快照)    : {len(gt)}")
    print(f"解析引号段          : {len(parsed)}")
    print(f"合并对齐成功        : {len(matched)} / {len(rows)} (未对齐 {len(rows)-len(matched)})")
    print(f"整体准确率(含群众)  : {overall_c}/{len(matched)} = {out['overall_accuracy_incl_crowd']:.1%}")
    print(f"具名角色准确率      : {named_c}/{len(named)} = {out['named_accuracy']:.1%}")
    print(f"30个难点用例        : {hard_c}/{len(hard)} = {out['hard_cases_accuracy']:.1%}")
    if conf:
        print("\n--- 具名角色混淆(错误) ---")
        for k, v in conf.most_common():
            print(f"  {v:>2}x  {k}")
    print(f"\n详细结果: {OUT}")


if __name__ == "__main__":
    main()
