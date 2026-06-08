"""Decisive test: forward re-ranker with FULL-NOVEL profiles.

Learn the forward address profile P(term|speaker) from the whole-novel gold snapshot
(rich; 琴纱月 has enough data), then apply the margin-gated re-ranker to a fresh seg2
parse. Does adequate profile data let it fix the 4-person opener without regressing?
"""
from __future__ import annotations
import copy, json, math, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "BookVoiceParser"))
sys.path.insert(0, str(REPO / "tools"))
from book_voice_parser import BatchConfig, parse_novel  # noqa
from book_voice_parser.schema import SegmentEx  # noqa
from book_voice_parser.address_term_backcheck import (  # noqa
    _extract_address_terms, _term_targets_speaker, _known_name_aliases, SKIP_SPEAKERS, GENERIC_SPEAKERS,
)
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa

a2c = {}
for c, al in ROLE_HINTS.items():
    if isinstance(al, (list, tuple)):
        for a in al:
            a2c[a] = c
canon = lambda x: a2c.get(x, x)
CROWD = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白", "其他", ""}


def norm(s):
    s = re.sub(r"[「」『』]", "", str(s or ""))
    return "".join(s.split()).replace("彷佛", "仿佛").replace("姊", "姐")


def learn_forward(segs, conf_min=0.0):
    aliases = _known_name_aliases(segs)
    counts = defaultdict(Counter); totals = Counter()
    for s in segs:
        sp = str(s.speaker or "")
        if sp in SKIP_SPEAKERS or sp in GENERIC_SPEAKERS:
            continue
        if float(s.confidence or 0) < conf_min:
            continue
        for term in _extract_address_terms(s.text, aliases, include_generic=True):
            if _term_targets_speaker(term, sp):
                continue
            counts[sp][term] += 1; totals[sp] += 1
    vocab = {t for c in counts.values() for t in c}
    return counts, totals, aliases, vocab


def rerank(segs, counts, totals, aliases, vocab, margin, min_support=3):
    V = max(1, len(vocab))
    logp = lambda term, X: math.log((counts[X][term] + 0.5) / (totals[X] + 0.5 * V))
    flips = []
    for i, s in enumerate(segs):
        cur = str(s.speaker or "")
        if cur in SKIP_SPEAKERS:
            continue
        terms = _extract_address_terms(s.text, aliases, include_generic=True)
        if not terms:
            continue
        cands = []
        for c in list(s.candidates or []) + [cur]:
            c = str(c or "")
            if not c or c in SKIP_SPEAKERS or c in GENERIC_SPEAKERS or totals[c] < min_support:
                continue
            if any(_term_targets_speaker(t, c) for t in terms):
                continue
            if c not in cands:
                cands.append(c)
        if cur not in cands or len(cands) < 2:
            continue
        score = {X: sum(logp(t, X) for t in terms) for X in cands}
        best = max(score, key=score.get)
        if best != cur and (score[best] - score[cur]) >= margin:
            flips.append((i, cur, best, score[best] - score[cur], terms))
            s.speaker = best
    return flips


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")
    # 1) forward profiles from full-novel gold snapshot
    snap = json.loads((SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json").read_text(encoding="utf-8"))["segments"]
    gold = [SegmentEx(speaker=s.get("speaker") or "旁白", text=s.get("text") or "", confidence=0.99,
                      attribution_type="implicit", quote_id=s.get("quote_id") or "") for s in snap]
    counts, totals, aliases, vocab = learn_forward(gold)
    print(f"金标正向画像: {len([k for k,v in totals.items() if v>=3])} 个角色(≥3); 纱月称呼数={totals['琴纱月']}")
    print(f"  P(甘织|纱月)~{counts['琴纱月']['甘织']}/{totals['琴纱月']}, P(甘织|香穗)~{counts['小柳香穗']['甘织']}/{totals['小柳香穗']}")

    # 2) fresh seg2 parse (block OFF) and rerank
    cfg = BatchConfig(base_url="https://apihub.agnes-ai.com/v1", api_key=os.environ["AGNES_API_KEY"],
                      model="agnes-2.0-flash", batch_size=8, max_tokens=5000, temperature=0.0,
                      timeout=180, context_chars=320, output_mode="compact", disable_thinking=True)
    off = parse_novel((SAMP / "muli4_seg2_sample.txt").read_text(encoding="utf-8"), role_hints=ROLE_HINTS,
                      batch_llm_config=cfg, narrator="甘织玲奈子", return_result=True,
                      include_narration=False, review_threshold=0.7, enable_block_review=False).segments
    gt = json.loads((SAMP / "muli4_seg2_groundtruth.json").read_text(encoding="utf-8"))["segments"]

    def sc(segs):
        nc = nt = 0
        for i in range(min(len(segs), len(gt))):
            if norm(segs[i].text) != norm(gt[i]["text"]):
                continue
            g = canon(gt[i]["speaker"])
            if not CROWD(g):
                nt += 1; nc += int(g == canon(segs[i].speaker))
        return nc, nt

    def opener(segs):
        return "  ".join(("✅" if canon(gt[j]["speaker"]) == canon(segs[j].speaker) else "❌") + f"[{j}]{segs[j].speaker}" for j in range(4))

    nc, nt = sc(off)
    print(f"\nbaseline(off): 具名 {nc}/{nt}={nc/max(1,nt):.1%}  开场: {opener(off)}")
    for m in [2.5, 3.5]:
        segs = copy.deepcopy(off)
        flips = rerank(segs, counts, totals, aliases, vocab, margin=m)
        nc, nt = sc(segs)
        print(f"\nmargin={m}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  翻转 {len(flips)}  开场: {opener(segs)}")
        for (i, old, new, d, terms) in flips:
            g = canon(gt[i]["speaker"]) if i < len(gt) and norm(segs[i].text) == norm(gt[i]["text"]) else None
            mark = "✅" if g == canon(new) else ("❌" if g == canon(old) else "·")
            print(f"   {mark} [{i}] {old}->{new} (Δ{d:.1f}, {terms}) 真值={g} | {segs[i].text[:24]}")


if __name__ == "__main__":
    main()
