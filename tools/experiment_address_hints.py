"""A/B the address-fingerprint hints in block review (option 2).

Isolated: one underlying parse per sample (block OFF), capturing the learned address-term
table; then apply block review with vs without those hints. Scored against authoritative GT.
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
                      return_result=True, include_narration=False, review_threshold=0.7,
                      enable_block_review=False)
    cleaned = normalize_text(raw)
    quotes = extract_quotes(cleaned)
    terms = (res.stats.get("address_term_backcheck") or {}).get("terms") or {}
    return res.segments, quotes, cleaned, terms


def review(off, quotes, cleaned, hints):
    segs = copy.deepcopy(off)
    aliases = AliasRegistry.from_role_hints(ROLE_HINTS)
    apply_block_review(segs, quotes, cleaned, cfg(), narrator="甘织玲奈子",
                       role_hints=list(ROLE_HINTS.keys()), aliases=aliases, address_hints=hints)
    return segs


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


def score_index(segs, gtsegs):
    nc = nt = sc = st = 0
    for i in range(min(len(segs), len(gtsegs))):
        if norm(segs[i].text) != norm(gtsegs[i]["text"]):
            continue
        g = canon(gtsegs[i]["speaker"]); p = canon(segs[i].speaker)
        if g in SIS: st += 1; sc += int(g == p)
        if not CROWD(g):
            nt += 1; nc += int(g == p)
    return nc, nt, sc, st


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")

    print("=== 样本1 first-hour ===", flush=True)
    t = (SAMP / "muli4_original_125697_utf8.txt").read_text(encoding="utf-16")
    raw1 = t[t.find("第四卷 序章"):t.find("是为了欢迎我回来", t.find("第四卷 序章")) + len("是为了欢迎我回来") + 80].strip()
    off1, q1, c1, terms1 = parse_off(raw1)
    print(f"  学得称呼指纹 {len(terms1)} 条: " + ", ".join(f"{k}->{v['speaker']}({v['score']})" for k, v in list(terms1.items())[:10]), flush=True)
    gt1, cnt1 = s1_gt()
    for label, hints in [("无指纹", None), ("有指纹", terms1)]:
        nc, nt, sc, st = score_text(review(off1, q1, c1, hints), gt1, cnt1)
        print(f"  {label}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {sc}/{st}={sc/max(1,st):.1%}", flush=True)

    print("\n=== 样本2 seg2 ===", flush=True)
    off2, q2, c2, terms2 = parse_off((SAMP / "muli4_seg2_sample.txt").read_text(encoding="utf-8"))
    print(f"  学得称呼指纹 {len(terms2)} 条: " + ", ".join(f"{k}->{v['speaker']}({v['score']})" for k, v in list(terms2.items())[:12]), flush=True)
    gt2 = json.loads((SAMP / "muli4_seg2_groundtruth.json").read_text(encoding="utf-8"))["segments"]
    for label, hints in [("无指纹", None), ("有指纹", terms2)]:
        segs = review(off2, q2, c2, hints)
        nc, nt, sc, st = score_index(segs, gt2)
        opening = "  ".join(("✅" if canon(gt2[j]["speaker"]) == canon(segs[j].speaker) else "❌") + f"[{j}]{segs[j].speaker}" for j in range(4))
        print(f"  {label}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {sc}/{st}={sc/max(1,st):.1%}", flush=True)
        print(f"     开场(真值 纱月/紫阳花/香穗/纱月): {opening}", flush=True)


if __name__ == "__main__":
    main()
