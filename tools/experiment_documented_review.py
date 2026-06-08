"""Verification: does the project's documented review pass (route_to_batch_llm)
actually help, when measured on FULL named accuracy (not just the 30 pre-checked cases)?

We run the real shipped review on our cached baseline parse and score net change with
authoritative GT, to test the doc's conclusion "复核无正向提升" against the user's intuition.
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
from book_voice_parser.schema import SegmentEx  # noqa: E402
from book_voice_parser import route_to_batch_llm  # noqa: E402
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402

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


class Cfg:
    base_url = "https://apihub.agnes-ai.com/v1"
    api_key = os.getenv("AGNES_API_KEY", "")
    model = "agnes-2.0-flash"
    max_tokens = 4096
    temperature = 0.0
    timeout = 180


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


def score(segs, gt, snap_cnt, label):
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
    print(f"[{label:<28}] 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {sc}/{st}={sc/max(1,st):.1%}")
    return nc, nt


def main():
    if not Cfg.api_key:
        raise SystemExit("AGNES_API_KEY not set")
    gt, snap_cnt = build_gt()
    raw = json.loads((SAMP / "_agnes_parse_cache.json").read_text(encoding="utf-8"))["segments"]

    base = [SegmentEx.model_validate(d) for d in raw]
    score(base, gt, snap_cnt, "baseline (no review)")

    # Documented review: target the genuinely-uncertain (implicit-type) segments by
    # lowering their confidence below threshold, then run the shipped batch review.
    review_in = [SegmentEx.model_validate(d) for d in raw]
    n_targets = 0
    for s in review_in:
        at = s.attribution_type
        at = at.value if hasattr(at, "value") else at
        if at in ("implicit", "unknown", "latent"):
            s.confidence = min(float(s.confidence or 0.5), 0.45)
            n_targets += 1
    print(f"[info] 复核目标(隐式/不确定)数: {n_targets}", flush=True)
    reviewed, stats = route_to_batch_llm(review_in, Cfg(), threshold=0.7, batch_size=8, narrator="甘织玲奈子")
    print(f"[info] 复核统计: reviewed={stats.get('reviewed')} corrected={stats.get('corrected')} "
          f"confirmed={stats.get('confirmed')} blocked={stats.get('blocked')}", flush=True)
    score(reviewed, gt, snap_cnt, "documented review (route_to_batch_llm)")
    print("\n对照: block decoder（结构化块复核）= 具名 84.2% / 姐妹 91.5%; 场景感知合并 = 85.9% / 87.9%")


if __name__ == "__main__":
    main()
