"""Prototype + A/B for the forward-likelihood address re-ranker.

For lines containing address terms, score each candidate by P(term | candidate) learned
from the parse itself (forward profile), and override the current speaker only when the
forward evidence overwhelmingly favors another candidate (large log-likelihood margin).
This resolves the inverse-lookup frequency bias.

Isolated, API-light: parse each sample once (block OFF) and run the offline re-rank with
a few margin settings. Prints every flip (right/wrong) and scores vs authoritative GT.
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
from book_voice_parser.address_term_backcheck import (  # noqa
    _extract_address_terms, _term_targets_speaker, _known_name_aliases,
    SKIP_SPEAKERS, GENERIC_SPEAKERS,
)
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


def learn_forward(segs, conf_min=0.9):
    aliases = _known_name_aliases(segs)
    counts = defaultdict(Counter)
    totals = Counter()
    for s in segs:
        sp = str(s.speaker or "")
        if sp in SKIP_SPEAKERS or sp in GENERIC_SPEAKERS:
            continue
        if float(s.confidence or 0) < conf_min:
            continue
        for term in _extract_address_terms(s.text, aliases, include_generic=True):
            if _term_targets_speaker(term, sp):
                continue
            counts[sp][term] += 1
            totals[sp] += 1
    vocab = {t for c in counts.values() for t in c}
    return counts, totals, aliases, vocab


def rerank(segs, counts, totals, aliases, vocab, margin, min_support=3):
    V = max(1, len(vocab))
    def logp(term, X):
        return math.log((counts[X][term] + 0.5) / (totals[X] + 0.5 * V))
    flips = []
    for i, s in enumerate(segs):
        cur = str(s.speaker or "")
        if cur in SKIP_SPEAKERS:
            continue
        terms = [t for t in _extract_address_terms(s.text, aliases, include_generic=True)]
        terms = [t for t in terms if any(totals[X] >= min_support for X in totals)]  # keep all; filter later per cand
        if not terms:
            continue
        # candidate set: named candidates with a learned profile, excluding term-targets
        cands = []
        for c in (s.candidates or []) + [cur]:
            c = str(c or "")
            if not c or c in SKIP_SPEAKERS or c in GENERIC_SPEAKERS:
                continue
            if totals[c] < min_support:
                continue
            if any(_term_targets_speaker(t, c) for t in terms):
                continue  # can't address self
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


def cfg():
    return BatchConfig(base_url="https://apihub.agnes-ai.com/v1", api_key=os.environ["AGNES_API_KEY"],
                       model="agnes-2.0-flash", batch_size=8, max_tokens=5000, temperature=0.0,
                       timeout=180, context_chars=320, output_mode="compact", disable_thinking=True)


def parse_off(raw):
    res = parse_novel(raw, role_hints=ROLE_HINTS, batch_llm_config=cfg(), narrator="甘织玲奈子",
                      return_result=True, include_narration=False, review_threshold=0.7,
                      enable_block_review=False)
    return res.segments


def s1_gt():
    snap = json.loads((SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json").read_text(encoding="utf-8"))["segments"]
    reg = json.loads((SAMP / "muli4_part001_first_hour_bookmark_regression.json").read_text(encoding="utf-8"))
    gt = {}
    for s in snap:
        sp = str(s.get("speaker") or ""); t = norm(s.get("text", ""))
        if sp and sp != "旁白" and t:
            gt.setdefault(t, sp)
    for c in reg["corrected_segments"]:
        gt[norm(snap[int(c["index"])].get("text", ""))] = str(c["expected_speaker"])
    cnt = Counter(norm(s.get("text", "")) for s in snap if s.get("speaker") and s.get("speaker") != "旁白")
    return gt, cnt


def score_text(segs, gt, cnt):
    pc = Counter(norm(s.text) for s in segs)
    nc = nt = 0
    for s in segs:
        k = norm(s.text)
        if k not in gt:
            continue
        g = canon(gt[k])
        if pc[k] == 1 and cnt.get(k, 0) == 1 and not CROWD(g):
            nt += 1; nc += int(g == canon(s.speaker))
    return nc, nt


def score_index(segs, gtsegs):
    nc = nt = 0
    for i in range(min(len(segs), len(gtsegs))):
        if norm(segs[i].text) != norm(gtsegs[i]["text"]):
            continue
        g = canon(gtsegs[i]["speaker"])
        if not CROWD(g):
            nt += 1; nc += int(g == canon(segs[i].speaker))
    return nc, nt


def gt_for(seg_text, gtmap):
    return gtmap.get(norm(seg_text))


def run_sample(name, off, gt_lookup, margins):
    counts, totals, aliases, vocab = learn_forward(off)
    print(f"  [{name}] 学得 {len(totals)} 个角色的正向画像; vocab={len(vocab)}", flush=True)
    base_nc, base_nt = gt_lookup["score"](off)
    print(f"  baseline: 具名 {base_nc}/{base_nt}={base_nc/max(1,base_nt):.1%}")
    for m in margins:
        segs = copy.deepcopy(off)
        flips = rerank(segs, counts, totals, aliases, vocab, margin=m)
        nc, nt = gt_lookup["score"](segs)
        good = sum(1 for (i, old, new, d, terms) in flips if gt_lookup["gt_idx"](i, segs) == canon(new))
        bad = sum(1 for (i, old, new, d, terms) in flips if gt_lookup["gt_idx"](i, segs) not in (None, canon(new)) and gt_lookup["gt_idx"](i, segs) == canon(old))
        print(f"  margin={m}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  | 翻转 {len(flips)} (修对≈{good} 改错≈{bad})")
        for (i, old, new, d, terms) in flips[:8]:
            g = gt_lookup["gt_idx"](i, segs)
            mark = "✅" if g == canon(new) else ("❌" if g == canon(old) else "·")
            print(f"      {mark} [{i}] {old}->{new} (Δ{d:.1f}, 称呼{terms}) 真值={g}")


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")

    print("=== 样本2 seg2 (全量真值) ===", flush=True)
    off2 = parse_off((SAMP / "muli4_seg2_sample.txt").read_text(encoding="utf-8"))
    gt2 = json.loads((SAMP / "muli4_seg2_groundtruth.json").read_text(encoding="utf-8"))["segments"]
    lk2 = {"score": lambda segs: score_index(segs, gt2),
           "gt_idx": lambda i, segs: canon(gt2[i]["speaker"]) if i < len(gt2) and norm(segs[i].text) == norm(gt2[i]["text"]) else None}
    run_sample("seg2", off2, lk2, margins=[2.0, 3.0, 4.0])

    print("\n=== 样本1 first-hour ===", flush=True)
    t = (SAMP / "muli4_original_125697_utf8.txt").read_text(encoding="utf-16")
    raw1 = t[t.find("第四卷 序章"):t.find("是为了欢迎我回来", t.find("第四卷 序章")) + len("是为了欢迎我回来") + 80].strip()
    off1 = parse_off(raw1)
    gt1, cnt1 = s1_gt()
    lk1 = {"score": lambda segs: score_text(segs, gt1, cnt1),
           "gt_idx": lambda i, segs: canon(gt1.get(norm(segs[i].text))) if gt1.get(norm(segs[i].text)) else None}
    run_sample("s1", off1, lk1, margins=[2.0, 3.0, 4.0])


if __name__ == "__main__":
    main()
