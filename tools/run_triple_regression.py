"""Three-sample regression: block_review OFF vs ON on seg1/seg2/seg3.

Isolation: one underlying parse per sample (block OFF), then apply block review to a
deep copy. seg3 uses acceptable-sets (a prediction is correct if it matches any acceptable
speaker, e.g. the interchangeable 藤村/清水 pair).
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


def parse_off_on(raw):
    res = parse_novel(raw, role_hints=ROLE_HINTS, batch_llm_config=cfg(), narrator="甘织玲奈子",
                      return_result=True, include_narration=False, review_threshold=0.7,
                      enable_block_review=False)
    off = res.segments
    cleaned = normalize_text(raw)
    quotes = extract_quotes(cleaned)
    on = copy.deepcopy(off)
    if len(quotes) == len(on):
        apply_block_review(on, quotes, cleaned, cfg(), narrator="甘织玲奈子",
                           role_hints=list(ROLE_HINTS.keys()), aliases=AliasRegistry.from_role_hints(ROLE_HINTS))
    else:
        print(f"  [warn] quotes({len(quotes)})!=segments({len(on)})")
    return off, on


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


def index_scorer(gtsegs):
    def score(segs):
        nc = nt = sc = st = 0
        for i in range(min(len(segs), len(gtsegs))):
            if norm(segs[i].text) != norm(gtsegs[i]["text"]):
                continue
            acc = {canon(x) for x in (gtsegs[i].get("acceptable") or [gtsegs[i]["speaker"]])}
            g = canon(gtsegs[i]["speaker"]); p = canon(segs[i].speaker)
            ok = p in acc
            if g in SIS: st += 1; sc += int(ok)
            if not CROWD(g):
                nt += 1; nc += int(ok)
        return nc, nt, sc, st
    return score


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")
    raw1, score1 = s1()
    jobs = [
        ("seg1 first-hour", raw1, score1),
        ("seg2", (SAMP / "muli4_seg2_sample.txt").read_text(encoding="utf-8"),
         index_scorer(json.loads((SAMP / "muli4_seg2_groundtruth.json").read_text(encoding="utf-8"))["segments"])),
        ("seg3", (SAMP / "muli4_seg3_sample.txt").read_text(encoding="utf-8"),
         index_scorer(json.loads((SAMP / "muli4_seg3_groundtruth.json").read_text(encoding="utf-8"))["segments"])),
    ]
    rows = []
    for name, raw, score in jobs:
        print(f"=== {name} ===", flush=True)
        off, on = parse_off_on(raw)
        for label, segs in [("OFF", off), ("ON ", on)]:
            nc, nt, sc, st = score(segs)
            print(f"  block_review {label}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹/2人 {sc}/{st}={sc/max(1,st):.1%}", flush=True)
            rows.append((name, label.strip(), nc / max(1, nt), sc / max(1, st)))
    print("\n=== 汇总（具名准确率 OFF→ON）===")
    for name in ["seg1 first-hour", "seg2", "seg3"]:
        off = next(r[2] for r in rows if r[0] == name and r[1] == "OFF")
        on = next(r[2] for r in rows if r[0] == name and r[1] == "ON")
        print(f"  {name:<16} {off:.1%} → {on:.1%}  (Δ{(on-off)*100:+.1f})")


if __name__ == "__main__":
    main()
