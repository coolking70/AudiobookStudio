#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性重建第五卷 part_005 修正快照（2026-06-16）。

输入：
  - 原优化快照（外部）：生成 part_005 时实际使用的 task_snapshot（segments 3001-3750）。
  - docs/samples/第五卷/角色清单.json：规范名↔别名表（脚本块标签归一用）。
  - docs/samples/第五卷/muli_5_part_005_review_flags.json：单遍第二意见 + 人工复核 resolution。

两层修正（与正文 README §part_005 一致）：
  1) 确定性：群组/剧本格式节切成逐行说话人段，标签按角色清单归一。
     （part_005 经探测无脚本块，本层 1:1 直通，不增段。）
  2) 人工复核：把 review_flags 中 resolution=="applied" 的改判按 quote_id 落到对应段。

输出：docs/samples/第五卷/muli_5_part_005_corrected_snapshot.json
用法：python tools/build_muli5_part005_snapshot.py  [--snapshot 路径]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "BookVoiceParser"))
from book_voice_parser.script_block import is_script_block, split_script_block_text  # noqa: E402
from book_voice_parser.alias_registry import AliasRegistry  # noqa: E402

OUT = REPO / "docs" / "samples" / "第五卷"
DEFAULT_SNAP = Path(r"I:\code\aitts\text\task_snapshot_segments_2026-06-12_optimized.json")
PART = slice(3000, 3750)  # 0-based segments 3001-3750


def _reset_audit(d: dict) -> None:
    d.update({"_needs_review": False, "_suspicious": False, "_suspicious_reason": "",
              "_llm_skipped": False, "_skip_reason": "", "_audit_tier": 0,
              "_audit_priority": None, "_audit_hint": ""})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAP), help="原优化快照路径（外部）")
    args = ap.parse_args()

    full = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    part = full["segments"][PART]
    roster = json.loads((OUT / "角色清单.json").read_text(encoding="utf-8"))
    reg = AliasRegistry.from_role_hints(roster)
    canon = lambda n: reg.canonicalize(n) or n  # noqa: E731

    # ① 脚本/群聊格式切分（确定性）
    out: list[dict] = []
    for s in part:
        if s["speaker"] in ("旁白", "未知") and is_script_block(s.get("text") or ""):
            pieces = split_script_block_text(s["text"])
            if sum(1 for n, _ in pieces if n) < 3:
                out.append(s); continue
            base = str(s.get("quote_id") or "seg"); sub = 0
            for name, content in pieces:
                sub += 1; d = json.loads(json.dumps(s))
                d.update({"quote_id": f"{base}_s{sub}", "text": content,
                          "context_before": "", "context_after": "", "addressee": None})
                _reset_audit(d)
                if name is None:
                    d.update({"speaker": "旁白", "attribution_type": "narrator", "confidence": 1.0,
                              "evidence": "群组/剧本格式：标签前的前言", "_confidence": "high",
                              "candidates": ["旁白"]})
                else:
                    sp = canon(name)
                    d.update({"speaker": sp, "attribution_type": "explicit_before", "confidence": 0.95,
                              "evidence": f"群组/剧本格式：行内显式说话人「{name}」→{sp}",
                              "_confidence": "high", "_evidence": f"行内显式说话人「{name}」",
                              "candidates": [sp], "candidate_sources": {sp: ["script_block"]}})
                out.append(d)
        else:
            out.append(s)

    # ② 人工复核改判（review_flags resolution==applied），按 quote_id 定位
    flags = json.loads((OUT / "muli_5_part_005_review_flags.json").read_text(encoding="utf-8"))
    idx2qid = {i + 1: str(part[i].get("quote_id")) for i in range(len(part))}
    apply = {idx2qid[it["part005_index_1based"]]: it["applied_speaker"]
             for it in flags["items"] if it.get("resolution") == "applied"}
    by_qid: dict[str, dict] = {}
    for s in out:
        by_qid.setdefault(str(s.get("quote_id")), s)
    applied = 0
    for qid, newsp in apply.items():
        s = by_qid.get(qid)
        if not s:
            continue
        old = s["speaker"]; s["speaker"] = newsp
        if newsp == "旁白":
            s["attribution_type"] = "narrator"; s["candidates"] = ["旁白"]
        else:
            s["attribution_type"] = "implicit"
            if newsp not in (s.get("candidates") or []):
                s["candidates"] = [newsp] + list(s.get("candidates") or [])
        s.update({"confidence": 0.9, "_confidence": "high", "_needs_review": False,
                  "evidence": f"人工复核(单遍第二意见)改判：{old}→{newsp}；" + str(s.get("evidence") or "")[:80]})
        applied += 1

    snap_out = dict(full)
    snap_out.update({
        "stage": "segments", "exportedAt": "2026-06-16_part005_corrected",
        "note": "第五卷 part_005（原 segments 3001-3750）+ 确定性脚本块切分 + 人工复核改判。",
        "segments": out,
    })
    (OUT / "muli_5_part_005_corrected_snapshot.json").write_text(
        json.dumps(snap_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"part_005 段数 {len(part)} → {len(out)}；人工改判落地 {applied}/{len(apply)}")
    print("写出 muli_5_part_005_corrected_snapshot.json")


if __name__ == "__main__":
    main()
