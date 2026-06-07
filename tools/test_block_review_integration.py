"""End-to-end test of the integrated block-review pass inside parse_novel.

Parses the original slice twice (block review off vs on) and scores both against
authoritative GT, to confirm the wired-in pass improves accuracy.
"""
from __future__ import annotations

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
from book_voice_parser import BatchConfig, parse_novel  # noqa: E402
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402

START, END = "第四卷 序章", "是为了欢迎我回来"
a2c = {}
for c, al in ROLE_HINTS.items():
    if isinstance(al, (list, tuple)):
        for a in al:
            a2c[a] = c
canon = lambda x: a2c.get(x, x)
SIS = {"甘织玲奈子", "甘织遥奈"}


def norm(s):
    s = re.sub(r"[「」『』]", "", str(s or ""))
    return "".join(s.split()).replace("彷佛", "仿佛").replace("姊", "姐")


def extract_slice():
    try:
        t = (SAMP / "muli4_original_125697_utf8.txt").read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        t = (SAMP / "muli4_original_125697_utf8.txt").read_text(encoding="utf-16")
    s = t.find(START); e = t.find(END, s); e = e + len(END) + 80
    return t[s:e].strip()


def build_gt():
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


def score(segs, gt, snap_cnt, label, stats=None):
    pcnt = Counter(norm(s.text) for s in segs)
    crowd = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白", "其他"}
    nc = nt = sc = st = 0
    for s in segs:
        k = norm(s.text)
        if k not in gt:
            continue
        g = canon(gt[k]); p = canon(s.speaker)
        if g in SIS:
            st += 1; sc += int(g == p)
        if pcnt[k] == 1 and snap_cnt.get(k, 0) == 1 and not crowd(g):
            nt += 1; nc += int(g == p)
    extra = f"  | block_review={stats}" if stats else ""
    print(f"[{label:<22}] 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {sc}/{st}={sc/max(1,st):.1%}{extra}")


def run(enable):
    cfg = BatchConfig(base_url="https://apihub.agnes-ai.com/v1", api_key=os.environ["AGNES_API_KEY"],
                      model="agnes-2.0-flash", batch_size=8, max_tokens=5000, temperature=0.0,
                      timeout=180, context_chars=320, output_mode="compact", disable_thinking=True)
    res = parse_novel(extract_slice(), role_hints=ROLE_HINTS, batch_llm_config=cfg, narrator="甘织玲奈子",
                      return_result=True, include_narration=False, review_threshold=0.7,
                      enable_block_review=enable)
    return res


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")
    gt, snap_cnt = build_gt()
    print("[run] parsing WITHOUT block review ...", flush=True)
    off = run(False)
    score(off.segments, gt, snap_cnt, "block_review OFF")
    print("[run] parsing WITH block review ...", flush=True)
    on = run(True)
    score(on.segments, gt, snap_cnt, "block_review ON", on.stats.get("block_review"))


if __name__ == "__main__":
    main()
