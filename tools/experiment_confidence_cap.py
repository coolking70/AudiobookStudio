"""Offline A/B experiment for Lever A: does capping BatchLLM's over-confident
`implicit` attributions re-enable the (already-present but dormant) fix_consistency
correction layer, and does it improve accuracy?

API-free: operates entirely on the cached parse (docs/samples/_agnes_parse_cache.json).

For each cap level we:
  1. reconstruct SegmentEx from the cached dicts,
  2. clamp confidence of attribution_type=implicit segments to the cap,
  3. re-run fix_consistency (scene-constraint / jump / two-person-repeat / alternation),
  4. re-score against authoritative ground truth (snapshot reviewed speaker, overridden by
     the regression `expected_speaker` on the 30 hard cases),
and report accuracy + how many speaker labels actually changed.
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
CACHE = SAMP / "_agnes_parse_cache.json"
SNAPSHOT = SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json"
REGRESSION = SAMP / "muli4_part001_first_hour_bookmark_regression.json"

sys.path.insert(0, str(REPO / "BookVoiceParser"))
from book_voice_parser.schema import SegmentEx, AttributionType  # noqa: E402
from book_voice_parser.consistency_fixer import fix_consistency  # noqa: E402

NARRATOR = "甘织玲奈子"


def norm(s: str) -> str:
    return "".join(str(s or "").split()).replace("彷佛", "仿佛").replace("姊", "姐")


def build_authoritative_gt() -> dict[str, str]:
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["segments"]
    reg = json.loads(REGRESSION.read_text(encoding="utf-8"))
    gt: dict[str, str] = {}
    for s in snap:
        sp = str(s.get("speaker") or "")
        t = norm(s.get("text", ""))
        if sp and sp != "旁白" and t:
            gt.setdefault(t, sp)
    for c in reg["corrected_segments"]:
        t = norm(snap[int(c["index"])].get("text", ""))
        gt[t] = str(c["expected_speaker"])
    return gt


def score(segments: list[SegmentEx], gt: dict[str, str], snap_cnt: Counter):
    """Unique-text robust scoring (collision-proof) against authoritative GT."""
    crowd = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白"}
    pcnt = Counter(norm(s.text) for s in segments)
    tot = cor = ntot = ncor = 0
    conf = Counter()
    for s in segments:
        nt = norm(s.text)
        if pcnt[nt] != 1 or snap_cnt.get(nt, 0) != 1 or nt not in gt:
            continue
        g = gt[nt]
        tot += 1
        cor += int(g == s.speaker)
        if not crowd(g):
            ntot += 1
            ncor += int(g == s.speaker)
            if g != s.speaker:
                conf[f"{g} -> {s.speaker}"] += 1
    return cor, tot, ncor, ntot, conf


def load_segments() -> list[SegmentEx]:
    raw = json.loads(CACHE.read_text(encoding="utf-8"))["segments"]
    return [SegmentEx.model_validate(d) for d in raw]


def run_variant(cap: float | None, gt, snap_cnt):
    segs = load_segments()
    before = [s.speaker for s in segs]
    capped = 0
    if cap is not None:
        for s in segs:
            at = s.attribution_type
            at_val = at.value if isinstance(at, AttributionType) else at
            if at_val == "implicit" and s.confidence > cap:
                s.confidence = cap
                capped += 1
        fix_consistency(segs, narrator=NARRATOR)
    changed = sum(1 for a, s in zip(before, segs) if a != s.speaker)
    cor, tot, ncor, ntot, conf = score(segs, gt, snap_cnt)
    return {
        "cap": cap, "capped_implicit": capped, "speaker_changed": changed,
        "overall": (cor, tot), "named": (ncor, ntot), "conf": conf,
    }


def main() -> None:
    gt = build_authoritative_gt()
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["segments"]
    snap_cnt = Counter(norm(s.get("text", "")) for s in snap if s.get("speaker") and s.get("speaker") != "旁白")

    # Diagnostic: do the real errors have the >=3-run shape _apply_alternation needs?
    base = load_segments()
    runs = []
    i = 0
    while i < len(base):
        j = i
        while j < len(base) and base[j].speaker == base[i].speaker:
            j += 1
        if j - i >= 3 and base[i].speaker not in {"旁白", "未知"}:
            runs.append((base[i].speaker, j - i))
        i = j
    print(f"[diag] >=3 同一说话人连续段（_apply_alternation 可触发的形状）: {len(runs)} 个 -> {runs[:8]}")

    print("\ncap        capped  changed  overall          named")
    results = []
    for cap in (None, 0.85, 0.80, 0.64, 0.50):
        r = run_variant(cap, gt, snap_cnt)
        results.append(r)
        oc, ot = r["overall"]; nc, nt = r["named"]
        label = "baseline" if cap is None else f"{cap:.2f}"
        print(f"{label:<10} {r['capped_implicit']:>5}  {r['speaker_changed']:>6}   "
              f"{oc}/{ot}={oc/max(1,ot):.1%}   {nc}/{nt}={nc/max(1,nt):.1%}")

    # Show confusion delta between baseline and best named variant.
    base_conf = results[0]["conf"]
    best = max(results[1:], key=lambda r: r["named"][0])
    print(f"\n最佳变体 cap={best['cap']}: named {best['named'][0]}/{best['named'][1]}")
    print("相比 baseline 的具名混淆变化：")
    keys = set(base_conf) | set(best["conf"])
    for k in sorted(keys, key=lambda x: -(base_conf.get(x, 0))):
        b, a = base_conf.get(k, 0), best["conf"].get(k, 0)
        if b != a:
            print(f"  {k:<28} {b} -> {a}")


if __name__ == "__main__":
    main()
