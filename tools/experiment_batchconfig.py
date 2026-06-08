"""Model/prompt-level A/B: re-parse the original slice under different BatchConfig
settings and score each against authoritative GT. Uses the real project pipeline
(parse_novel + Agnes BatchLLM). API is free; backoff handles rate limits.

Hypothesis: the residual errors are short fast back-and-forth turns; tighter local
context (larger context_chars) and smaller batches (fresher recent-speaker anchoring)
may help the model track the exchange.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "BookVoiceParser"))
sys.path.insert(0, str(REPO / "tools"))
from book_voice_parser import BatchConfig, parse_novel  # noqa: E402
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402

START_MARKER, END_MARKER = "第四卷 序章", "是为了欢迎我回来"
LING, HARU = "甘织玲奈子", "甘织遥奈"
SIS = {LING, HARU}

a2c = {}
for c, al in ROLE_HINTS.items():
    if isinstance(al, (list, tuple)):
        for a in al:
            a2c[a] = c
canon = lambda x: a2c.get(x, x)


def norm(s: str) -> str:
    return "".join(str(s or "").split()).replace("彷佛", "仿佛").replace("姊", "姐")


def extract_slice() -> str:
    try:
        text = (SAMP / "muli4_original_125697_utf8.txt").read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = (SAMP / "muli4_original_125697_utf8.txt").read_text(encoding="utf-16")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER, start)
    end = end + len(END_MARKER) + 80 if end >= 0 else min(len(text), start + 70000)
    return text[start:end].strip()


def build_gt():
    snap = json.loads((SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json").read_text(encoding="utf-8"))["segments"]
    reg = json.loads((SAMP / "muli4_part001_first_hour_bookmark_regression.json").read_text(encoding="utf-8"))
    gt = {}
    for s in snap:
        sp = str(s.get("speaker") or "")
        t = norm(s.get("text", ""))
        if sp and sp != "旁白" and t:
            gt.setdefault(t, sp)
    for c in reg["corrected_segments"]:
        gt[norm(snap[int(c["index"])].get("text", ""))] = str(c["expected_speaker"])
    snap_cnt = Counter(norm(s.get("text", "")) for s in snap if s.get("speaker") and s.get("speaker") != "旁白")
    return gt, snap_cnt


def score(parsed, gt, snap_cnt):
    crowd = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白"}
    pcnt = Counter(norm(s.get("text", "")) for s in parsed)
    nc = nt = sc = st = 0
    for s in parsed:
        k = norm(s.get("text", ""))
        if k not in gt:
            continue
        g = canon(gt[k]); p = canon(s.get("speaker", ""))
        if g in SIS:
            st += 1; sc += int(g == p)
        if pcnt[k] == 1 and snap_cnt.get(k, 0) == 1 and not crowd(g):
            nt += 1; nc += int(g == p)
    return nc, nt, sc, st


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")
    text = extract_slice()
    gt, snap_cnt = build_gt()
    base = dict(base_url="https://apihub.agnes-ai.com/v1", api_key=os.environ["AGNES_API_KEY"],
                model="agnes-2.0-flash", max_tokens=5000, temperature=0.0, timeout=180,
                output_mode="compact", disable_thinking=True)
    variants = [
        ("bs8 ctx320 (baseline)", dict(batch_size=8, context_chars=320)),
        ("bs4 ctx640",            dict(batch_size=4, context_chars=640)),
        ("bs2 ctx900",            dict(batch_size=2, context_chars=900)),
        ("bs6 ctx500",            dict(batch_size=6, context_chars=500)),
        ("bs8 ctx320 v1.5",       dict(batch_size=8, context_chars=320, model="agnes-1.5-flash")),
    ]
    rows = []
    for name, ov in variants:
        cfg_kw = {**base, **ov}
        cfg = BatchConfig(**cfg_kw)
        t0 = time.time()
        try:
            res = parse_novel(text, role_hints=ROLE_HINTS, batch_llm_config=cfg, narrator=LING,
                              return_result=True, include_narration=False, review_threshold=0.7)
            parsed = [s.model_dump(mode="json") for s in res.segments]
            nc, nt, sc, st = score(parsed, gt, snap_cnt)
            dur = time.time() - t0
            rows.append((name, nc, nt, sc, st, len(parsed), dur))
            print(f"{name:<26} 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {sc}/{st}={sc/max(1,st):.1%}  seg={len(parsed)} {dur:.0f}s", flush=True)
            (SAMP / f"_parse_{name.split()[0]}_{ov.get('model','v2')[-3:]}.json").write_text(
                json.dumps({"name": name, "segments": parsed}, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"{name:<26} ERROR {str(e)[:80]}", flush=True)

    print("\n=== 汇总 ===")
    for name, nc, nt, sc, st, n, dur in rows:
        print(f"{name:<26} 具名={nc/max(1,nt):.1%}  姐妹={sc/max(1,st):.1%}  ({dur:.0f}s)")


if __name__ == "__main__":
    main()
