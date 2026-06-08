"""MCP server exposing the dense-multi-person-scene speaker re-attribution step.

Lets any MCP-capable agent (Claude Desktop / Claude Code, etc.) perform the
dense-scene review using its OWN model — no strong-model API key is exposed.
The ground truth (if present) stays server-side and is only used to score
submitted attributions; it is never sent to the agent.

Tools:
  - list_dense_scenes(sample)        -> passages (tagged, no answer key) + roster + method
  - submit_attributions(sample, ...) -> apply + (if GT exists) score; writes a review file
  - list_samples()                   -> available samples

Run:  python tools/mcp_review_server.py        (stdio)
Wire into a client, e.g. Claude Code:
  claude mcp add audiobook-review -- /path/.venv/bin/python /path/tools/mcp_review_server.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"

# Single-task mode: when set (via --task-file), all tools operate on this one task
# (the live analysis task exported by the web UI) instead of the docs/samples dev files.
_TASK_FILE: Path | None = None
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "BookVoiceParser"))

try:
    from evaluate_agnes_bookmark_review import ROLE_HINTS
except Exception:
    ROLE_HINTS = {}

_A2C = {}
for _c, _al in ROLE_HINTS.items():
    if isinstance(_al, (list, tuple)):
        for _a in _al:
            _A2C[_a] = _c
canon = lambda x: _A2C.get(x, x)

DENSE_WINDOW = 4
DENSE_MIN = 3

METHOD = (
    "按真人读法判断每句对话说话人：\n"
    "1. 先读旁白确定每段【在场人物】与【登场顺序】（'A与B走过来'/'X介绍说有甲乙丙'/'X齐聚一堂'，先出现/先被点名者先开口）。\n"
    "2. 用点名/感谢/排除缩小【实际参与者】（'谢谢A、B、C'即三人在场参与；全程没被提及/没接话者多半未参与）。\n"
    "⚠️前文若已给出登场/介绍顺序，必须按该顺序分配首发与轮次，不要随机；只有完全无序时才默认交替。\n"
    "3. 逐句判断：续说 / 两人交替 / 第三人 / 多对一（两人轮流向一人搭话、该人以疑问回应）。无线索时默认按登场顺序交替。\n"
    "4. 校正：称呼/口癖（被称呼的'X'是受话人不是说话人；'喵'口癖→小柳香穗）、语气语域、语义邻接（问→答、提议→拒绝、递→接）、参与者集合内排除法。\n"
    "5. 【场景一致性】没有旁白明确写某人加入/离开时，说话人集合保持稳定、按既定轮次进行；⚠️不要引入旁白未提及、不在本场景的角色。\n"
    "6. 第一人称'我'说出口→叙述者甘织玲奈子；『』内心独白→发出者。\n"
    "7. 无名群体（一群摄影师/女生/起哄众人）统一用「群众·<群体名>」（例：群众·摄影师们、群众·女性们）；单个无名路人→其他；非台词→旁白。\n"
    "只用角色表里的规范名（群众·… 除外）。"
)


def _crowd(s: str) -> bool:
    s = s or ""
    return s.startswith("群众·") or s.startswith("厕所女生") or s in {
        "未知临时人物", "未知", "其他", "旁白", ""}


def _roster() -> list[str]:
    out = []
    for c, al in ROLE_HINTS.items():
        al = al if isinstance(al, (list, tuple)) else []
        out.append(c + (f"（别名：{'、'.join(map(str, al))}）" if al else ""))
    return out


def _stem(sample: str) -> str:
    sample = sample.strip()
    return sample if sample.startswith("muli4_") else f"muli4_{sample}"


def _load(sample: str):
    """Return (stem, raw_text, segments, gt_or_None).

    Task mode (--task-file): load the live task snapshot ({sourceText, segments});
    production has no ground truth (gt=None). Sample mode: dev files under docs/samples.
    """
    if _TASK_FILE is not None:
        task = json.loads(_TASK_FILE.read_text(encoding="utf-8"))
        raw = task.get("sourceText") or task.get("source_text") or ""
        parse = task.get("segments") or []
        return "current", raw, parse, None
    stem = _stem(sample)
    raw = (SAMP / f"{stem}_sample.txt").read_text(encoding="utf-8")
    parse = json.loads((SAMP / f"{stem}_parse.json").read_text(encoding="utf-8"))["segments"]
    gtf = SAMP / f"{stem}_groundtruth.json"
    gt = {s["i"]: s for s in json.loads(gtf.read_text(encoding="utf-8"))["segments"]} if gtf.exists() else None
    return stem, raw, parse, gt


def _corrections_path() -> Path:
    if _TASK_FILE is not None:
        return _TASK_FILE.with_suffix(".corrections.json")
    return SAMP / "_mcp_corrections.json"


def _quote_positions(raw: str, parse: list[dict]):
    cur, pos = 0, []
    for s in parse:
        t = s["text"]
        p = raw.find(t, cur)
        pos.append((p, p + len(t)) if p >= 0 else (-1, -1))
        if p >= 0:
            cur = p + len(t)
    return pos


def _dense_flags(parse: list[dict]):
    names = ["" if _crowd(s["speaker"]) else canon(s["speaker"]) for s in parse]
    flags = []
    for i in range(len(parse)):
        loc = {names[j] for j in range(max(0, i - DENSE_WINDOW), min(len(parse), i + DENSE_WINDOW + 1)) if names[j]}
        flags.append(len(loc) >= DENSE_MIN)
    return flags


mcp = FastMCP("audiobook-review")


@mcp.tool()
def list_samples() -> dict:
    """列出可复核的对象。任务模式下为当前分析任务（current）；否则为 docs/samples 下已解析样本。"""
    if _TASK_FILE is not None:
        return {"samples": ["current"], "mode": "task"}
    stems = sorted({p.name[:-len("_parse.json")] for p in SAMP.glob("*_parse.json")})
    return {"samples": [s.replace("muli4_", "") for s in stems], "mode": "samples"}


@mcp.tool()
def list_dense_scenes(sample: str) -> dict:
    """返回某样本中【密集多人场景】（局部≥3具名说话人）的待复核片段。

    每段是原文（旁白+对话），每句对话前有 【sample_i】 标记。返回不含任何答案，
    供智能体用自身模型按 method 重判。用 submit_attributions 回传结果。
    """
    stem, raw, parse, _ = _load(sample)
    sk = stem.replace("muli4_", "")
    pos = _quote_positions(raw, parse)
    dflags = _dense_flags(parse)
    # contiguous dense runs, ±2 context
    runs, i = [], 0
    while i < len(parse):
        if dflags[i]:
            j = i
            while j < len(parse) and dflags[j]:
                j += 1
            runs.append((max(0, i - 2), min(len(parse), j + 2)))
            i = j
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    passages, dense_tags = [], []
    for a, b in merged:
        if pos[a][0] < 0:
            continue
        ws = max(0, pos[a][0] - 180)
        we = (pos[b - 1][1] + 80) if pos[b - 1][1] > 0 else pos[a][1] + 1200
        win = raw[ws:we]
        for p, k in sorted([(pos[k][0], k) for k in range(a, b) if pos[k][0] >= 0], reverse=True):
            win = win[: p - ws] + f"【{sk}_{k}】" + win[p - ws:]
        for k in range(a, b):
            if dflags[k] and not _crowd(parse[k]["speaker"]):
                dense_tags.append(f"{sk}_{k}")
        passages.append(win.strip())
    return {
        "sample": sk,
        "method": METHOD,
        "roster": _roster(),
        "passages": passages,
        "dense_tags": dense_tags,
        "note": "为每个出现的【tag】给出规范名说话人；dense_tags 是评分目标（密集场景具名句）。",
    }


@mcp.tool()
def submit_attributions(sample: str, attributions: dict) -> dict:
    """回传 {标记: 说话人} 的归因结果。写入复核文件；若该样本有人工真值，则就密集场景句给出准确率。

    attributions: 形如 {"seg1_188": "小柳香穗", ...}。真值仅在服务器端用于打分，不外泄。
    """
    stem, raw, parse, gt = _load(sample)
    sk = stem.replace("muli4_", "")
    # persist corrections (tag -> speaker), keyed by segment index
    corrections = {}
    for tag, spk in attributions.items():
        m = re.match(rf"^{re.escape(sk)}_(\d+)$", str(tag))
        if m:
            corrections[m.group(1)] = str(spk)
    out_path = _corrections_path()
    out_path.write_text(json.dumps({"source": "mcp_review", "corrections": corrections},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"sample": sk, "applied": len(corrections), "review_file": str(out_path.name)}
    if gt is not None:
        dflags = _dense_flags(parse)
        nc = nt = changed_right = changed_wrong = 0
        wrong = []
        for i, seg in enumerate(parse):
            if not dflags[i] or _crowd(gt[i]["speaker"]):
                continue
            tag = f"{sk}_{i}"
            if tag not in attributions:
                continue
            acc = {canon(x) for x in (gt[i].get("acceptable") or [gt[i]["speaker"]])}
            pred = canon(str(attributions[tag]))
            nt += 1
            ok = pred in acc
            nc += int(ok)
            model_ok = canon(seg["speaker"]) in acc
            if ok and not model_ok:
                changed_right += 1
            if not ok and model_ok:
                changed_wrong += 1
            if not ok:
                wrong.append({"tag": tag, "your": str(attributions[tag]), "text": gt[i]["text"][:30]})
        result.update({
            "dense_scored": nt,
            "dense_accuracy": round(nc / max(1, nt), 4),
            "vs_baseline_fixed": changed_right,
            "vs_baseline_broke": changed_wrong,
            "remaining_wrong": wrong[:20],
        })
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Audiobook dense-scene review MCP server")
    ap.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8970)
    ap.add_argument("--task-file", type=Path, default=None,
                    help="当前分析任务快照（{sourceText, segments}）；给出后所有工具针对该任务而非 docs/samples")
    args = ap.parse_args()
    if args.task_file is not None:
        _TASK_FILE = args.task_file.expanduser().resolve()
    if args.transport in ("sse", "streamable-http"):
        # bind locally so the web UI can launch it and a client connects to http://host:port/sse
        mcp.settings.host = args.host
        mcp.settings.port = args.port
    mcp.run(transport=args.transport)
