"""Dual-sample regression: block_review OFF vs ON, on both authoritative samples,
with the scene-alias leak fix in place.

Isolation: parse each sample once (block review OFF) to get baseline segments, then
deep-copy and apply block review — so OFF vs ON share the same underlying parse (no
run-to-run variance), measuring purely the block-review delta.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "BookVoiceParser"))
sys.path.insert(0, str(REPO / "tools"))
from book_voice_parser import BatchConfig, parse_novel  # noqa
from book_voice_parser.block_review import apply_block_review  # noqa
from book_voice_parser.cleaner import normalize_text  # noqa
from book_voice_parser.quote_extractor import extract_quotes  # noqa
from book_voice_parser.alias_registry import AliasRegistry  # noqa
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa

a2c = {}
for c, al in ROLE_HINTS.items():
    if isinstance(al, (list, tuple)):
        for a in al:
            a2c[a] = c
canon = lambda x: a2c.get(x, x)
SIS = {"甘织玲奈子", "甘织遥奈"}
CROWD = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白", "其他", ""}


def norm(s):
    s = re.sub(r"[「」『』]", "", str(s or ""))
    return "".join(s.split()).replace("彷佛", "仿佛").replace("姊", "姐")


def cfg():
    return BatchConfig(base_url="https://apihub.agnes-ai.com/v1", api_key=os.environ["AGNES_API_KEY"],
                       model="agnes-2.0-flash", batch_size=8, max_tokens=5000, temperature=0.0,
                       timeout=180, context_chars=320, output_mode="compact", disable_thinking=True)


def parse_off_on(raw: str):
    """Return (off_segments, on_segments) sharing one underlying parse."""
    res = parse_novel(raw, role_hints=ROLE_HINTS, batch_llm_config=cfg(), narrator="甘织玲奈子",
                      return_result=True, include_narration=False, review_threshold=0.7,
                      enable_block_review=False)
    off = res.segments
    cleaned = normalize_text(raw)
    quotes = extract_quotes(cleaned)
    on = copy.deepcopy(off)
    if len(quotes) == len(on):
        aliases = AliasRegistry.from_role_hints(ROLE_HINTS)
        apply_block_review(on, quotes, cleaned, cfg(), narrator="甘织玲奈子",
                           role_hints=list(ROLE_HINTS.keys()), aliases=aliases)
    else:
        print(f"  [warn] quotes({len(quotes)}) != segments({len(on)}); block review skipped")
    return off, on


# ---- Sample 1 (first hour): authoritative GT by text ----
def sample1_raw():
    t = (SAMP / "muli4_original_125697_utf8.txt").read_text(encoding="utf-16")
    s = t.find("第四卷 序章"); e = t.find("是为了欢迎我回来", s); e = e + len("是为了欢迎我回来") + 80
    return t[s:e].strip()


def sample1_gt():
    snap = json.loads((SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json").read_text(encoding="utf-8"))["segments"]
    reg = json.loads((SAMP / "muli4_part001_first_hour_bookmark_regression.json").read_text(encoding="utf-8"))
    gt = {}
    for s in snap:
        sp = str(s.get("speaker") or ""); t = norm(s.get("text", ""))
        if sp and sp != "旁白" and t:
            gt.setdefault(t, sp)
    for c in reg["corrected_segments"]:
        gt[norm(snap[int(c["index"])].get("text", ""))] = str(c["expected_speaker"])
    snap_cnt = Counter(norm(s.get("text", "")) for s in snap if s.get("speaker") and s.get("speaker") != "旁白")
    return gt, snap_cnt


def score_by_text(segs, gt, snap_cnt):
    pcnt = Counter(norm(s.text) for s in segs)
    nc = nt = sc = st = 0
    for s in segs:
        k = norm(s.text)
        if k not in gt:
            continue
        g = canon(gt[k]); p = canon(s.speaker)
        if g in SIS:
            st += 1; sc += int(g == p)
        if pcnt[k] == 1 and snap_cnt.get(k, 0) == 1 and not CROWD(g):
            nt += 1; nc += int(g == p)
    return nc, nt, sc, st


# ---- Sample 2 (seg2): per-segment GT by index ----
def score_by_index(segs, gt_segments):
    """gt_segments: list of {i, speaker, text}. Align by index; sanity-check text."""
    nc = nt = sc = st = mism = 0
    n = min(len(segs), len(gt_segments))
    for i in range(n):
        if norm(segs[i].text) != norm(gt_segments[i]["text"]):
            mism += 1
            continue
        g = canon(gt_segments[i]["speaker"]); p = canon(segs[i].speaker)
        if g in SIS:
            st += 1; sc += int(g == p)
        if not CROWD(g):
            nt += 1; nc += int(g == p)
    return nc, nt, sc, st, mism


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")

    print("=== 样本1（first hour）===", flush=True)
    off1, on1 = parse_off_on(sample1_raw())
    gt1, sc1 = sample1_gt()
    for label, segs in [("OFF", off1), ("ON ", on1)]:
        nc, nt, s, t = score_by_text(segs, gt1, sc1)
        print(f"  block_review {label}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {s}/{t}={s/max(1,t):.1%}")

    print("\n=== 样本2（seg2, 全量人工真值）===", flush=True)
    off2, on2 = parse_off_on((SAMP / "muli4_seg2_sample.txt").read_text(encoding="utf-8"))
    gt2 = json.loads((SAMP / "muli4_seg2_groundtruth.json").read_text(encoding="utf-8"))["segments"]
    for label, segs in [("OFF", off2), ("ON ", on2)]:
        nc, nt, s, t, mism = score_by_index(segs, gt2)
        note = f"  (文本错位 {mism})" if mism else ""
        print(f"  block_review {label}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {s}/{t}={s/max(1,t):.1%}{note}")


if __name__ == "__main__":
    main()
