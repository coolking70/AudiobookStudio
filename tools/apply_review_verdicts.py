#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复核裁决回贴器（确定性，2026-06-16）。

吃强模型对复核包给出的逐行裁决，按 quote_id 落回 part 修正快照，并更新 review_flags 状态。
与 make_review_packet.py 配套，构成"确定性预处理 → 强模型只判语义 → 确定性回贴"闭环。

裁决文件每行：`<编号> <规范名|K|D>`（K=维持流水线, D=两可待定），允许行内多余内容，按前两段解析。

用法：python tools/apply_review_verdicts.py --part 5 --verdicts path/to/verdicts.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "samples" / "第五卷"
sys.path.insert(0, str(REPO / "BookVoiceParser"))
from book_voice_parser.alias_registry import AliasRegistry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--verdicts", required=True, help="强模型裁决文本（每行 编号 说话人/K/D）")
    args = ap.parse_args()
    n = args.part

    roster = json.loads((OUT / "角色清单.json").read_text(encoding="utf-8"))
    reg = AliasRegistry.from_role_hints(roster)
    idmap = json.loads((OUT / f"muli_5_part_00{n}_review_packet.idmap.json").read_text(encoding="utf-8"))
    flags = json.loads((OUT / f"muli_5_part_00{n}_review_flags.json").read_text(encoding="utf-8"))
    snap_p = OUT / f"muli_5_part_00{n}_corrected_snapshot.json"
    snap = json.loads(snap_p.read_text(encoding="utf-8"))

    # 解析裁决
    verdict: dict[str, str] = {}
    for line in Path(args.verdicts).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*\[?(\d+)\]?\s+(\S+)", line.strip())
        if m:
            verdict[m.group(1)] = m.group(2)

    by_qid = {str(s.get("quote_id")): s for s in snap["segments"]}
    applied = kept = deferred = miss = 0
    key = next(k for k in flags["items"][0] if k.endswith("index_1based"))
    for it in flags["items"]:
        ix = str(it[key])
        v = verdict.get(ix)
        if v is None:
            continue
        if v in ("K", "k", "维持", "keep"):
            it["resolution"] = "kept"; kept += 1
            continue
        if v in ("D", "d", "待定", "defer"):
            it["resolution"] = "deferred"; deferred += 1
            continue
        newsp = reg.canonicalize(v) or v
        qid = idmap.get(ix)
        seg = by_qid.get(str(qid))
        if not seg:
            miss += 1
            continue
        old = seg["speaker"]
        seg["speaker"] = newsp
        if newsp == "旁白":
            seg["attribution_type"] = "narrator"; seg["candidates"] = ["旁白"]
        else:
            seg["attribution_type"] = "implicit"
            if newsp not in (seg.get("candidates") or []):
                seg["candidates"] = [newsp] + list(seg.get("candidates") or [])
        seg["confidence"] = 0.9; seg["_confidence"] = "high"; seg["_needs_review"] = False
        seg["evidence"] = f"人工复核(强模型)改判：{old}→{newsp}；" + str(seg.get("evidence") or "")[:80]
        it["resolution"] = "applied"; it["applied_speaker"] = newsp
        applied += 1

    flags["resolution_summary"] = {"applied": applied, "kept": kept, "deferred": deferred}
    snap_p.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"muli_5_part_00{n}_review_flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"应用裁决：改判 {applied} ｜ 维持 {kept} ｜ 待定 {deferred} ｜ 未命中 {miss}")


if __name__ == "__main__":
    main()
