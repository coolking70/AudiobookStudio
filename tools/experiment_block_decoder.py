"""Prototype: block-level turn decoder (structured second-pass attribution).

Re-attributes dialogue in coherent context windows using a prompt that encodes the
human reading strategy:
  1. read the narration to find who is present and their ORDER OF APPEARANCE
     (introductions like "A与B/A和B走过来");
  2. anchor the first speaker from that order;
  3. for each line decide continue-same / alternate-two / third-party, defaulting to
     two-person alternation when there is no other cue;
  4. correct using 称呼 (vocatives/kinship), 语气/语域 (register), and 语义邻接 (Q->A,
     offer->reject, hand-over->take).

Same model as the baseline pipeline (agnes-2.0-flash) so this tests the METHOD.
Scored against authoritative GT and compared to the baseline cached parse.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "tools"))
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402

BASE_URL = "https://apihub.agnes-ai.com/v1"
MODEL = "agnes-2.0-flash"
LING, HARU = "甘织玲奈子", "甘织遥奈"
SIS = {LING, HARU}

a2c = {}
for c, al in ROLE_HINTS.items():
    if isinstance(al, (list, tuple)):
        for a in al:
            a2c[a] = c
canon = lambda x: a2c.get(x, x)


def norm(s: str) -> str:
    s = re.sub(r"[「」『』]", "", str(s or ""))
    return "".join(s.split()).replace("彷佛", "仿佛").replace("姊", "姐")


def roster_text() -> str:
    lines = []
    for c, al in ROLE_HINTS.items():
        al = al if isinstance(al, (list, tuple)) else []
        lines.append(c + (f"（别名：{'、'.join(map(str, al))}）" if al else ""))
    return "\n".join(lines)


SYSTEM = (
    "你是中文轻小说的说话人归属专家。像真人读小说那样判断每句对话由谁说出。"
    "叙述是第一人称，叙述者“我”=甘织玲奈子。"
)

PROMPT_TMPL = """下面是连续的一段原文（旁白+对话混排），每句对话前有【qNNN】标记。

【角色表】（请用规范名，不要用别名）：
{roster}

【判断方法，请严格按此推理】：
1. 先读旁白，确定这一段【在场人物】和他们的【登场顺序】——留意“A与B/A和B走了过来”这类并列引入，先出现的人通常先开口。
2. 用登场顺序【锚定第一句】的说话人，定下交替基准。
3. 逐句判断它是【与上句同一人续说】【两人交替】还是【有新的人加入/某人退出】。没有其他线索时，默认两人按登场顺序交替。
4. 用以下证据【校正】：
   - 称呼：出现“姊姊/姐姐”是妹妹甘织遥奈在叫姐姐甘织玲奈子→该句说话人是遥奈；“甘织同学/玲奈亲”等是别人叫玲奈子。
   - 语气/语域：玲奈子说话慌乱、自我贬低、内心戏多；遥奈毒舌、冷静、爱吐槽。
   - 语义邻接：提问→回答、提议→拒绝、递东西→接话，多为两人交替；一个人连续做一件事（如翻看相簿逐张点评）可能是同一人续说。
5. 第一人称“我”说出口的话归甘织玲奈子；用『』括起的内心独白/回想，仍归发出该想法的角色（通常是玲奈子）。
6. 无名路人（如厕所里闲聊的陌生女生）归“其他”；若某【q】其实不是说出口的话则归“旁白”。

【本段原文】：
{block}

只输出一个 JSON 对象，键是本段出现的每个 q 编号，值是规范名。不要任何解释。例如：
{{"q012":"甘织玲奈子","q013":"甘织遥奈"}}"""


def call_llm(prompt: str, api_key: str) -> str:
    payload = {"model": MODEL, "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 1500}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempt = 0
    with httpx.Client(timeout=120, trust_env=False) as cli:
        while True:
            r = cli.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
            if r.status_code == 429 and attempt < 6:
                time.sleep(min(60, 5 * 2 ** attempt)); attempt += 1; continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


def parse_json(content: str) -> dict:
    content = re.sub(r"^```(json)?|```$", "", content.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def make_blocks(annotated: str, qids: list[str], per: int = 12, lookback: int = 3):
    """Windows of `per` new quotes with `lookback` overlap; each window text spans from
    the lookback quote's tag to just before the next window's first new quote."""
    pos = {q: annotated.find(f"【{q}】") for q in qids}
    blocks = []
    i = 0
    while i < len(qids):
        start_i = max(0, i - lookback)
        end_i = min(len(qids), i + per)
        s = pos[qids[start_i]]
        e = pos[qids[end_i]] if end_i < len(qids) else len(annotated)
        new_ids = qids[i:end_i]
        blocks.append((annotated[s:e], new_ids))
        i = end_i
    return blocks


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
    snap_cnt = Counter(norm(s.get("text", "")) for s in snap if s.get("speaker") and s.get("speaker") != "旁白")
    return gt, snap_cnt


def score(pred_by_qid, qmap, gt, snap_cnt, label):
    qtext_cnt = Counter(norm(t) for t in qmap.values())
    crowd = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白", "其他"}
    nc = nt = sc = st = 0
    mis = Counter()
    for qid, qt in qmap.items():
        k = norm(qt)
        if k not in gt:
            continue
        g = canon(gt[k]); p = canon(pred_by_qid.get(qid, "?"))
        if g in SIS:
            st += 1; sc += int(g == p)
        if qtext_cnt[k] == 1 and snap_cnt.get(k, 0) == 1 and not crowd(g):
            nt += 1
            if g == p:
                nc += 1
            else:
                mis[f"{g} -> {p}"] += 1
    print(f"[{label}] 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {sc}/{st}={sc/max(1,st):.1%}")
    return nc, nt, sc, st, mis


def main():
    api_key = os.getenv("AGNES_API_KEY")
    if not api_key:
        raise SystemExit("AGNES_API_KEY not set")
    inp = json.loads((Path("/tmp/subagent_input.json")).read_text(encoding="utf-8"))
    annotated, qmap = inp["annotated"], inp["qmap"]
    qids = sorted(qmap, key=lambda q: int(q[1:]))
    gt, snap_cnt = build_gt()

    blocks = make_blocks(annotated, qids)
    print(f"对话块数: {len(blocks)} (覆盖 {len(qids)} 句)")
    pred = {}
    rtext = roster_text()
    for bi, (btext, new_ids) in enumerate(blocks):
        out = parse_json(call_llm(PROMPT_TMPL.format(roster=rtext, block=btext), api_key))
        for q in new_ids:
            if q in out:
                pred[q] = out[q]
        print(f"  block {bi+1}/{len(blocks)} -> {len(out)} preds", flush=True)

    Path("/tmp/block_decoder_output.json").write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    print()
    _, _, _, _, mis = score(pred, qmap, gt, snap_cnt, "block-decoder")
    print("\n具名错误:")
    for k, v in mis.most_common():
        print(f"  {v:>2}x {k}")


if __name__ == "__main__":
    main()
