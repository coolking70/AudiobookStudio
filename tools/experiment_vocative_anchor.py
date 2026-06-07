"""Offline experiment for Lever B: vocative / relationship anchoring in 2-person
sister blocks, with and without alternation-fill. API-free (cached parse).

Sister relationship anchor: 遥奈 is the younger sibling, so a quote that uses the
kinship vocative 姊姊/姐姐 as a term of address is spoken BY 遥奈 (addressing 玲奈子).

Variants:
  B0 baseline      : cached parse as-is
  B1 vocative-only : inside a 2-person {玲奈子,遥奈} block, force vocative-bearing turns -> 遥奈
  B2 anchor+altern : B1 anchors + explicit_* anchors, then fill gaps between consecutive
                     anchors by strict alternation parity

Scored against authoritative GT (snapshot reviewed, overridden by regression expected),
two-sided alias-canonicalized, on the sister-pair segments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "BookVoiceParser"))
sys.path.insert(0, str(REPO / "tools"))
from book_voice_parser.schema import SegmentEx  # noqa: E402
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402

LING, HARU = "甘织玲奈子", "甘织遥奈"
SIS = {LING, HARU}
VOCATIVE = ("姊姊", "姐姐")


def norm(s: str) -> str:
    return "".join(str(s or "").split()).replace("彷佛", "仿佛").replace("姊", "姐")


a2c = {}
for c, al in ROLE_HINTS.items():
    if isinstance(al, (list, tuple)):
        for a in al:
            a2c[a] = c
canon = lambda x: a2c.get(x, x)


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
    return gt


def load():
    raw = json.loads((SAMP / "_agnes_parse_cache.json").read_text(encoding="utf-8"))["segments"]
    return [SegmentEx.model_validate(d) for d in raw]


def find_blocks(segs):
    """Maximal runs (len>=4) of consecutive segments whose speaker is in the sister pair."""
    blocks = []
    i = 0
    while i < len(segs):
        if canon(segs[i].speaker) in SIS:
            j = i
            while j < len(segs) and canon(segs[j].speaker) in SIS:
                j += 1
            if j - i >= 4:
                blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def apply_vocative(segs, blocks):
    for lo, hi in blocks:
        for k in range(lo, hi):
            if any(v in segs[k].text for v in VOCATIVE):
                segs[k].speaker = HARU


def apply_alternation_fill(segs, blocks):
    """Anchors = vocative(->遥奈) and explicit_* (trust current). Between consecutive
    anchors, fill by strict alternation if parity is consistent with both ends."""
    for lo, hi in blocks:
        anchors = {}
        for k in range(lo, hi):
            at = segs[k].attribution_type
            at = at.value if hasattr(at, "value") else at
            if any(v in segs[k].text for v in VOCATIVE):
                anchors[k] = HARU
            elif at in ("explicit_before", "explicit_after", "explicit_near"):
                anchors[k] = canon(segs[k].speaker)
        keys = sorted(anchors)
        for a, b in zip(keys, keys[1:]):
            gap = b - a
            # alternation consistent only if endpoints have opposite parity-implied speakers
            if anchors[a] == anchors[b] and gap % 2 == 0:
                who = anchors[a]; other = (SIS - {who}).pop()
                for k in range(a + 1, b):
                    segs[k].speaker = other if (k - a) % 2 == 1 else who
            elif anchors[a] != anchors[b] and gap % 2 == 1:
                who = anchors[a]; other = anchors[b]
                for k in range(a + 1, b):
                    segs[k].speaker = who if (k - a) % 2 == 0 else other


def score_sisters(segs, gt):
    cor = tot = 0
    for s in segs:
        k = norm(s.text)
        if k not in gt:
            continue
        g = canon(gt[k])
        if g not in SIS:
            continue
        p = canon(s.speaker)
        if p not in SIS and g in SIS:
            # still count as attempted sister turn
            pass
        tot += 1
        cor += int(g == p)
    return cor, tot


def score_named(segs, gt):
    from collections import Counter
    snap = json.loads((SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json").read_text(encoding="utf-8"))["segments"]
    snap_cnt = Counter(norm(s.get("text", "")) for s in snap if s.get("speaker") and s.get("speaker") != "旁白")
    pcnt = Counter(norm(s.text) for s in segs)
    crowd = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白"}
    cor = tot = 0
    for s in segs:
        k = norm(s.text)
        if pcnt[k] != 1 or snap_cnt.get(k, 0) != 1 or k not in gt:
            continue
        g = canon(gt[k])
        if crowd(g):
            continue
        tot += 1
        cor += int(g == canon(s.speaker))
    return cor, tot


def main():
    gt = build_gt()
    base = load()
    blocks = find_blocks(base)
    print(f"检测到的姐妹双人块: {blocks}")

    for name, fn in [("B0 baseline", None), ("B1 vocative-only", "voc"), ("B2 anchor+altern", "alt")]:
        segs = load()
        bl = find_blocks(segs)
        if fn == "voc":
            apply_vocative(segs, bl)
        elif fn == "alt":
            apply_vocative(segs, bl)
            apply_alternation_fill(segs, bl)
        sc, st = score_sisters(segs, gt)
        nc, nt = score_named(segs, gt)
        print(f"{name:<18} 姐妹 {sc}/{st}={sc/max(1,st):.1%}   全具名 {nc}/{nt}={nc/max(1,nt):.1%}")


if __name__ == "__main__":
    main()
