"""Evaluate any OpenAI-compatible model on the authoritative samples (full BatchLLM
pipeline + block review), scored against ground truth, split by scene size.

Usage:
  MODEL_BASE_URL=https://omnitok.xyz/v1 MODEL_NAME=gpt-5.5 \
  MODEL_API_KEY=sk-... .venv/bin/python tools/eval_external_model.py
"""
from __future__ import annotations
import json, os, re, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "BookVoiceParser"))
sys.path.insert(0, str(REPO / "tools"))
from book_voice_parser import BatchConfig, parse_novel  # noqa
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa

a2c = {}
for c, al in ROLE_HINTS.items():
    if isinstance(al, (list, tuple)):
        for a in al:
            a2c[a] = c
canon = lambda x: a2c.get(x, x)
SIS = {"甘织玲奈子", "甘织遥奈"}
CROWD = lambda x: (x or "").startswith("厕所女生") or (x or "").startswith("群众·") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白", "其他", ""}


def norm(s):
    s = re.sub(r"[「」『』]", "", str(s or ""))
    return "".join(s.split()).replace("彷佛", "仿佛").replace("姊", "姐")


def cfg():
    return BatchConfig(
        base_url=os.environ["MODEL_BASE_URL"], api_key=os.environ.get("MODEL_API_KEY", ""),
        model=os.environ["MODEL_NAME"],
        batch_size=int(os.environ.get("MODEL_BATCH_SIZE", "8")),
        max_tokens=int(os.environ.get("MODEL_MAX_TOKENS", "5000")),
        temperature=0.0, timeout=int(os.environ.get("MODEL_TIMEOUT", "240")),
        context_chars=320, output_mode="compact",
        disable_thinking=os.environ.get("MODEL_DISABLE_THINKING", "1") == "1",
    )


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


def scene_size(gtsegs, i, w=4):
    s = set()
    for j in range(max(0, i - w), min(len(gtsegs), i + w + 1)):
        sp = gtsegs[j]["speaker"]
        if not CROWD(sp):
            s.add(canon(sp))
    return len(s)


def score_index_split(segs, gtsegs):
    simple = [0, 0]; dense = [0, 0]
    for i in range(min(len(segs), len(gtsegs))):
        if norm(segs[i].text) != norm(gtsegs[i]["text"]):
            continue
        acc = {canon(x) for x in (gtsegs[i].get("acceptable") or [gtsegs[i]["speaker"]])}
        g = canon(gtsegs[i]["speaker"])
        if CROWD(g):
            continue
        ok = canon(segs[i].speaker) in acc
        b = simple if scene_size(gtsegs, i) <= 2 else dense
        b[1] += 1; b[0] += int(ok)
    return simple, dense


def run(raw):
    res = parse_novel(raw, role_hints=ROLE_HINTS, batch_llm_config=cfg(), narrator="甘织玲奈子",
                      return_result=True, include_narration=False, review_threshold=0.7, enable_block_review=True)
    return res.segments


def main():
    for v in ("MODEL_BASE_URL", "MODEL_NAME"):
        if not os.environ.get(v):
            raise SystemExit(f"set {v}")
    print(f"模型: {os.environ['MODEL_NAME']} @ {os.environ['MODEL_BASE_URL']}\n", flush=True)
    # sample 1 (text-based GT, overall named only)
    t = (SAMP / "muli4_seg1_sample.txt").read_text(encoding="utf-8")
    print("=== seg1 ===", flush=True)
    segs = run(t)
    gt1, cnt1 = s1_gt()
    nc, ntot = score_text(segs, gt1, cnt1)
    print(f"  具名(文本口径): {nc}/{ntot}={nc/max(1,ntot):.1%}")
    # seg2/seg3 (per-index GT, scene-size split)
    for stem in ["muli4_seg2", "muli4_seg3"]:
        print(f"=== {stem.replace('muli4_','')} ===", flush=True)
        segs = run((SAMP / f"{stem}_sample.txt").read_text(encoding="utf-8"))
        gtsegs = json.loads((SAMP / f"{stem}_groundtruth.json").read_text(encoding="utf-8"))["segments"]
        si, de = score_index_split(segs, gtsegs)
        tot = [si[0] + de[0], si[1] + de[1]]
        print(f"  具名 {tot[0]}/{tot[1]}={tot[0]/max(1,tot[1]):.1%} | 简单≤2人 {si[0]}/{si[1]}={si[0]/max(1,si[1]):.1%} | 密集≥3人 {de[0]}/{de[1]}={de[0]/max(1,de[1]):.1%}")
    print("\n对照 agnes-2.0-flash: 简单≤2人 ~96.7% / 密集≥3人 ~82% / 整体 ~93%")


if __name__ == "__main__":
    main()
