"""Test the slice-boundary hypothesis: parse seg1+seg2 as one continuous passage and
re-score the seg2 portion (esp. the 4-person opening) against seg2 ground truth.
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
CROWD = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白", "其他", ""}


def norm(s):
    s = re.sub(r"[「」『』]", "", str(s or ""))
    return "".join(s.split()).replace("彷佛", "仿佛").replace("姊", "姐")


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")
    t = (SAMP / "muli4_original_125697_utf8.txt").read_text(encoding="utf-16")
    s1 = t.find("第四卷 序章")
    m = t.find("是为了欢迎我回来", s1)
    seg2_start = m + len("是为了欢迎我回来") + 80
    seg2_len = len((SAMP / "muli4_seg2_sample.txt").read_text(encoding="utf-8"))
    target = seg2_start + seg2_len
    i = t.find("\n \n", target)
    seg2_end = i if 0 <= i < target + 1200 else target
    continuous = t[s1:seg2_end].strip()
    (SAMP / "muli4_concat_sample.txt").write_text(continuous, encoding="utf-8")
    print(f"拼接样本: {len(continuous)} 字 (seg1+seg2 连续)", flush=True)

    cfg = BatchConfig(base_url="https://apihub.agnes-ai.com/v1", api_key=os.environ["AGNES_API_KEY"],
                      model="agnes-2.0-flash", batch_size=8, max_tokens=5000, temperature=0.0,
                      timeout=180, context_chars=320, output_mode="compact", disable_thinking=True)
    res = parse_novel(continuous, role_hints=ROLE_HINTS, batch_llm_config=cfg, narrator="甘织玲奈子",
                      return_result=True, include_narration=False, review_threshold=0.7, enable_block_review=True)
    segs = [s.model_dump(mode="json") for s in res.segments]
    (SAMP / "muli4_concat_parse.json").write_text(json.dumps({"segments": segs}, ensure_ascii=False), encoding="utf-8")

    # build pred-by-text (whole continuous); use last occurrence walking to map seg2 GT
    gt = json.loads((SAMP / "muli4_seg2_groundtruth.json").read_text(encoding="utf-8"))["segments"]
    # map each parsed seg text -> list of speakers in order
    by_text = {}
    for sseg in segs:
        by_text.setdefault(norm(sseg["text"]), []).append(sseg["speaker"])
    # consume seg2 GT in order (seg2 portion appears once, after seg1)
    used = Counter()
    nc = nt = sc = st = 0
    opening = []
    for j, g in enumerate(gt):
        k = norm(g["text"])
        lst = by_text.get(k, [])
        idx = used[k]
        pred = lst[idx] if idx < len(lst) else None
        used[k] += 1
        if pred is None:
            continue
        G = canon(g["speaker"]); P = canon(pred)
        if G in SIS:
            st += 1; sc += int(G == P)
        if not CROWD(G):
            nt += 1; nc += int(G == P)
        if j < 4:
            opening.append((j, g["speaker"], pred, g["text"][:22]))
    print(f"\nseg2 部分（连续上下文）: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {sc}/{st}={sc/max(1,st):.1%}")
    print(f"对照 seg2 独立解析: 具名 94.8% / 姐妹 97.1%（人工真值 95.5% 全段）")
    print("\n4 人开场 [0-3]（真值 vs 连续解析）:")
    for j, gtsp, pred, txt in opening:
        flag = "✅" if canon(gtsp) == canon(pred) else "❌"
        print(f"  {flag} [{j}] 真值={gtsp:<8} 解析={pred:<8} 「{txt}」")


if __name__ == "__main__":
    main()
