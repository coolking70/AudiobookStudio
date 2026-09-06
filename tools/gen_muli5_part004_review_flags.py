#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成第五卷 part_004 单遍第二意见 review_flags（2026-06-16）。

跑 single-pass（整文 + 角色表）第二意见，与流水线 speaker 比对，
**只对分歧段**产出 review_flags.json 骨架（resolution 留 "deferred"），
供人工逐条对原文填 applied/kept/deferred。

schema 严格对齐 part_003 的 muli_5_part_003_review_flags.json，
确保 tools/build_muli5_part004_snapshot.py 能消费。

输入：
  - 原优化快照（外部）：segments 2251-3000（part_004）。
  - docs/samples/第五卷/原文_part004.txt：part_004 正文。
  - docs/samples/第五卷/角色清单.json：规范名↔别名。
输出：
  - docs/samples/第五卷/muli_5_part_004_review_flags.json（建议未人工复核时勿提交）。

用法（先 set AGNES_API_KEY，见 docs/samples/SAMPLE_WORKFLOW.md §0）：
    python tools/gen_muli5_part004_review_flags.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "BookVoiceParser"))
from book_voice_parser.single_pass_attributor import SinglePassAttributor  # noqa: E402
from book_voice_parser.batch_llm_attributor import BatchConfig  # noqa: E402
from book_voice_parser.schema import QuoteSpan  # noqa: E402

OUT = REPO / "docs" / "samples" / "第五卷"
DEFAULT_SNAP = Path(r"I:\code\aitts\text\task_snapshot_segments_2026-06-12_optimized.json")
PART = slice(2250, 3000)  # 0-based segments 2251-3000

AGNES_URL = "https://apihub.agnes-ai.com/v1/chat/completions"
AGNES_MODEL = "agnes-2.0-flash"

# 不计分的群众/旁白：单遍不重问这些。
CROWD = {"未知", "未知临时人物", "旁白", "其他", ""}


def _is_crowd(sp: str) -> bool:
    sp = sp or ""
    return sp.startswith(("群众·", "厕所女生")) or sp in CROWD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAP), help="原优化快照路径（外部）")
    ap.add_argument("--chunk-size", type=int, default=200)
    args = ap.parse_args()

    api_key = os.environ.get("AGNES_API_KEY")
    if not api_key:
        raise SystemExit("未设置 AGNES_API_KEY —— 见 docs/samples/SAMPLE_WORKFLOW.md §0")

    full = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    part = full["segments"][PART]
    raw = (OUT / "原文_part004.txt").read_text(encoding="utf-8")
    roster = json.loads((OUT / "角色清单.json").read_text(encoding="utf-8"))

    # role_aliases：owner 型妈妈条目取 aliases 子表，普通条目用别名列表
    role_aliases = {
        c: (al if isinstance(al, (list, tuple)) else al.get("aliases", []))
        for c, al in roster.items()
    }
    # a2c：别名→规范名（含规范名自身），用于把单遍输出归一到规范名
    a2c = {c: c for c in roster}
    for c, al in roster.items():
        names = al if isinstance(al, (list, tuple)) else al.get("aliases", [])
        for a in names:
            a2c[a] = c

    def canon(s: str) -> str:
        s = (s or "").strip()
        if s in a2c:
            return a2c[s]
        for x, c in a2c.items():
            if x and x in s:
                return c
        return s

    # 构造待归因 quotes（仅具名对话段，排除群众/旁白/narrator）
    quotes, qid2idx = [], {}
    n_dialogue = 0
    for i, s in enumerate(part):
        if _is_crowd(s.get("speaker", "")) or s.get("attribution_type") == "narrator":
            continue
        n_dialogue += 1
        qid = str(s.get("quote_id") or f"p4_{i+1}")
        txt = s.get("text", "")
        quotes.append(QuoteSpan(quote_id=qid, text=txt, start=0, end=len(txt),
                                context_before="", context_after=""))
        qid2idx[qid] = i  # 0-based part 内索引

    print(f"part_004: {len(part)} 段，具名对话段 {n_dialogue} 个，跑单遍第二意见…", flush=True)
    cfg = BatchConfig(base_url=AGNES_URL.rsplit("/chat/completions", 1)[0],
                      api_key=api_key, model=AGNES_MODEL, timeout=300)
    sp = SinglePassAttributor(cfg, full_text=raw, chunk_size=args.chunk_size)

    def on_progress(done, total):
        print(f"  单遍归因 {done}/{total}", flush=True)

    res = sp.attribute(quotes, role_hints=list(roster.keys()),
                       narrator="甘织玲奈子", role_aliases=role_aliases, on_progress=on_progress)

    # 分歧即入 flags（suggest-only：仅记录，不改写流水线）
    items = []
    for qid, a in res.items():
        i = qid2idx.get(qid)
        if i is None:
            continue
        pipe_sp = part[i].get("speaker", "")
        sug = canon(str(getattr(a, "speaker", "") or ""))
        if not sug or _is_crowd(sug):
            continue
        if sug != canon(pipe_sp):
            items.append({
                "part004_index_1based": i + 1,
                "quote_id": qid,
                "pipeline_speaker": pipe_sp,
                "singlepass_suggestion": sug,
                "text": part[i].get("text", ""),
                "resolution": "deferred",  # 人工待填
            })

    items.sort(key=lambda it: it["part004_index_1based"])
    out = {
        "part": "muli_5_part_004",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "method": "single-pass(整文+角色表) 第二意见 vs 流水线，suggest-only 未自动改写",
        "note": "分歧供人工复核：单遍召回高(part_001 实测98%)但精度~51%，含双人轮换相位翻转误报，"
                "请逐条判断后再决定是否改入 corrected_snapshot。 "
                "| resolution 取值：applied(改入快照,需带 applied_speaker) / kept(维持流水线) / deferred(两可待定)。",
        "dialogue_segments_checked": n_dialogue,
        "flags": len(items),
        "items": items,
    }
    out_p = OUT / "muli_5_part_004_review_flags.json"
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {out_p.name}：{len(items)} 条分歧（具名对话 {n_dialogue} 段）")
    print("下一步：逐条对原文填 resolution；applied 必须带 applied_speaker。")


if __name__ == "__main__":
    main()
