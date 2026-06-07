"""Forward re-ranker with VOCATIVE-only extraction.

Only direct-address vocatives (X，… / …，X / X你) count toward the fingerprint and the
candidate scoring; third-person mentions (找X / 选X / 安慰X) are excluded. Profiles from
full-novel gold; applied to a fresh seg2 parse. Goal: keep the [0] opener fix while
dropping the mention-driven bad flips ([123]/[46]).
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
from book_voice_parser.address_term_backcheck import _term_targets_speaker, _known_name_aliases, SKIP_SPEAKERS, GENERIC_SPEAKERS  # noqa
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa

a2c = {}
for c, al in ROLE_HINTS.items():
    if isinstance(al, (list, tuple)):
        for a in al:
            a2c[a] = c
canon = lambda x: a2c.get(x, x)
CROWD = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白", "其他", ""}

LEFT_BD = set("。！？!?，,、；;：:「『（(【「]\n 　")
RIGHT_VOC = set("，,、。！？!?…～~」』】）) 　")
FOLLOW = ("你", "您", "妳")
MENTION = ("找", "选", "给", "安慰", "叫", "问", "陪", "帮", "和", "跟", "向", "对", "提", "说", "想",
           "看", "等", "替", "为", "像", "比", "请", "让", "被", "约", "骂", "夸", "怪", "谢", "告诉",
           "喜欢", "讨厌", "支持", "遇到", "认识", "记得", "忘", "卷", "带", "成为", "当作", "关于", "代替", "的")


def norm(s):
    s = re.sub(r"[「」『』]", "", str(s or ""))
    return "".join(s.split()).replace("彷佛", "仿佛").replace("姊", "姐")


def vocative_terms(text, aliases):
    text = str(text or "")
    out = []
    for a in sorted({x for x in aliases if len(x) >= 2}, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(a, start)
            if i < 0:
                break
            j = i + len(a)
            left = text[i - 1] if i > 0 else ""
            left3 = text[max(0, i - 3):i]
            right = text[j] if j < len(text) else ""
            is_mention = any(left3.endswith(v) for v in MENTION)
            clause_start = (i == 0) or (left in LEFT_BD)
            is_voc = (not is_mention) and (
                (clause_start and (right in RIGHT_VOC or right in FOLLOW or right == "")) or
                (left in ("，", "、")) or
                (right in FOLLOW)
            )
            if is_voc and not any(a.startswith(o) or o.startswith(a) for o in out):
                out.append(a)
            start = j
    return out


def learn_forward(segs):
    aliases = _known_name_aliases(segs)
    counts = defaultdict(Counter); totals = Counter()
    for s in segs:
        sp = str(s.speaker or "")
        if sp in SKIP_SPEAKERS or sp in GENERIC_SPEAKERS:
            continue
        for term in vocative_terms(s.text, aliases):
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
        terms = vocative_terms(s.text, aliases)
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
    snap = json.loads((SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json").read_text(encoding="utf-8"))["segments"]
    gold = [SegmentEx(speaker=s.get("speaker") or "旁白", text=s.get("text") or "", confidence=0.99,
                      attribution_type="implicit", quote_id=s.get("quote_id") or "") for s in snap]
    counts, totals, aliases, vocab = learn_forward(gold)
    print(f"金标呼语画像: 纱月呼语数={totals['琴纱月']} (甘织={counts['琴纱月']['甘织']}), 香穗={totals['小柳香穗']} (甘织={counts['小柳香穗']['甘织']}, 玲奈亲={counts['小柳香穗']['玲奈亲']})")

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

    # show vocatives extracted for the opener lines
    print("\n开场各句抽到的呼语:")
    for j in range(4):
        print(f"  [{j}] {vocative_terms(off[j].text, aliases)}  | {off[j].text[:26]}")

    nc, nt = sc(off)
    print(f"\nbaseline(off): 具名 {nc}/{nt}={nc/max(1,nt):.1%}  开场: {opener(off)}")
    for m in [2.5, 3.5]:
        segs = copy.deepcopy(off)
        flips = rerank(segs, counts, totals, aliases, vocab, margin=m)
        nc, nt = sc(segs)
        good = sum(1 for (i, o, n, d, t) in flips if i < len(gt) and norm(segs[i].text) == norm(gt[i]["text"]) and canon(gt[i]["speaker"]) == canon(n))
        bad = sum(1 for (i, o, n, d, t) in flips if i < len(gt) and norm(segs[i].text) == norm(gt[i]["text"]) and canon(gt[i]["speaker"]) == canon(o))
        print(f"\nmargin={m}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  翻转{len(flips)}(对{good}/错{bad})  开场: {opener(segs)}")
        for (i, o, n, d, t) in flips:
            g = canon(gt[i]["speaker"]) if i < len(gt) and norm(segs[i].text) == norm(gt[i]["text"]) else None
            mark = "✅" if g == canon(n) else ("❌" if g == canon(o) else "·")
            print(f"   {mark} [{i}] {o}->{n} (Δ{d:.1f}, {t}) 真值={g} | {segs[i].text[:22]}")


if __name__ == "__main__":
    main()
