"""A/B: block review default (re-decode) vs conservative (审核+仅证据改判), 3 samples.

Same OFF baseline per sample; apply block review in each mode to a copy. Reports named +
2-person accuracy and, for seg3, the specific flip cases the human flagged.
"""
from __future__ import annotations
import copy, json, os, re, sys
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


def parse_off(raw):
    res = parse_novel(raw, role_hints=ROLE_HINTS, batch_llm_config=cfg(), narrator="甘织玲奈子",
                      return_result=True, include_narration=False, review_threshold=0.7, enable_block_review=False)
    return res.segments, extract_quotes(normalize_text(raw)), normalize_text(raw)


def review(off, quotes, cleaned, conservative):
    segs = copy.deepcopy(off)
    if len(quotes) == len(segs):
        apply_block_review(segs, quotes, cleaned, cfg(), narrator="甘织玲奈子",
                           role_hints=list(ROLE_HINTS.keys()), aliases=AliasRegistry.from_role_hints(ROLE_HINTS),
                           conservative=conservative)
    return segs


def s1():
    t = (SAMP / "muli4_original_125697_utf8.txt").read_text(encoding="utf-16")
    raw = t[t.find("第四卷 序章"):t.find("是为了欢迎我回来", t.find("第四卷 序章")) + len("是为了欢迎我回来") + 80].strip()
    snap = json.loads((SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json").read_text(encoding="utf-8"))["segments"]
    reg = json.loads((SAMP / "muli4_part001_first_hour_bookmark_regression.json").read_text(encoding="utf-8"))
    gt = {}
    for s in snap:
        sp = str(s.get("speaker") or ""); tx = norm(s.get("text", ""))
        if sp and sp != "旁白" and tx:
            gt.setdefault(tx, sp)
    for c in reg["corrected_segments"]:
        gt[norm(snap[int(c["index"])].get("text", ""))] = str(c["expected_speaker"])
    cnt = Counter(norm(s.get("text", "")) for s in snap if s.get("speaker") and s.get("speaker") != "旁白")

    def score(segs):
        pc = Counter(norm(s.text) for s in segs)
        nc = nt = sc = st = 0
        for s in segs:
            k = norm(s.text)
            if k not in gt:
                continue
            g = canon(gt[k]); p = canon(s.speaker)
            if g in SIS: st += 1; sc += int(g == p)
            if pc[k] == 1 and cnt.get(k, 0) == 1 and not CROWD(g):
                nt += 1; nc += int(g == p)
        return nc, nt, sc, st
    return raw, score


def idx_scorer(gtsegs):
    def score(segs):
        nc = nt = sc = st = 0
        for i in range(min(len(segs), len(gtsegs))):
            if norm(segs[i].text) != norm(gtsegs[i]["text"]):
                continue
            acc = {canon(x) for x in (gtsegs[i].get("acceptable") or [gtsegs[i]["speaker"]])}
            g = canon(gtsegs[i]["speaker"]); ok = canon(segs[i].speaker) in acc
            if g in SIS: st += 1; sc += int(ok)
            if not CROWD(g):
                nt += 1; nc += int(ok)
        return nc, nt, sc, st
    return score


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")
    raw1, sc1 = s1()
    gt3 = json.loads((SAMP / "muli4_seg3_groundtruth.json").read_text(encoding="utf-8"))["segments"]
    jobs = [
        ("seg1", raw1, sc1, None),
        ("seg2", (SAMP / "muli4_seg2_sample.txt").read_text(encoding="utf-8"),
         idx_scorer(json.loads((SAMP / "muli4_seg2_groundtruth.json").read_text(encoding="utf-8"))["segments"]), None),
        ("seg3", (SAMP / "muli4_seg3_sample.txt").read_text(encoding="utf-8"), idx_scorer(gt3), gt3),
    ]
    for name, raw, score, gt3seg in jobs:
        print(f"=== {name} ===", flush=True)
        off, quotes, cleaned = parse_off(raw)
        on = review(off, quotes, cleaned, conservative=False)
        cons = review(off, quotes, cleaned, conservative=True)
        for label, segs in [("OFF      ", off), ("ON 默认  ", on), ("ON 保守  ", cons)]:
            nc, nt, s, t = score(segs)
            print(f"  {label}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  2人 {s}/{t}={s/max(1,t):.1%}", flush=True)
        if gt3seg:
            print("  seg3 关注段 (真值 / 默认 / 保守):")
            for i in [44, 161, 163, 164, 172, 277, 278]:
                g = gt3seg[i]["speaker"]
                print(f"    [{i}] 真值={g:<8} 默认={on[i].speaker:<8} 保守={cons[i].speaker:<8} "
                      f"{'✅' if canon(cons[i].speaker)==canon(g) else '❌'}")


if __name__ == "__main__":
    main()
