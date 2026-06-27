#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""说话人复核包生成器（确定性，2026-06-16）。

把强模型复核所需的"依据材料"用确定性流水线预先抽好，使强模型**只花 token 在语义判断**：
不必读整卷快照/原文、不必自己跑单遍、不必复述流程。

输入：某 part 的 review_flags（单遍第二意见分歧）+ 源快照 + 角色清单。
输出：
  - docs/samples/第五卷/muli_5_part_00N_review_packet.md  （交给强模型复制粘贴）
      顶部：别名表 + 判定优先级 + 输出格式（一次性）
      正文：每条 flag 一块 = 紧邻上下文(±W) + 候选/在场 + 流水线/第二意见两方意见
  - muli_5_part_00N_review_packet.idmap.json  （编号 → quote_id，供 apply 回贴）

用法：
  python tools/make_review_packet.py --part 5 [--window 2]
  python tools/make_review_packet.py --snapshot path/to/parse.json --audit path/to/audit.json \\
    --roster path/to/roster.json --output-prefix sample_name [--window 2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "samples" / "第五卷"
SNAP = Path(r"I:\code\aitts\text\task_snapshot_segments_2026-06-12_optimized.json")


def _alias_line(roster: dict) -> str:
    parts = []
    for canon, al in roster.items():
        names = al if isinstance(al, list) else al.get("aliases", [])
        names = [a for a in names if a and a != canon]
        if names:
            parts.append(f"{canon}←{'/'.join(names)}")
    return "；".join(parts)


def _alias_map(roster: dict) -> dict[str, str]:
    aliases = {}
    for canon, value in roster.items():
        aliases[str(canon)] = str(canon)
        names = value if isinstance(value, list) else value.get("aliases", [])
        for name in names:
            if name:
                aliases[str(name)] = str(canon)
    return aliases


def _compact_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _review_candidates(segment: dict, limit: int = 12) -> list[str]:
    generic = {
        "旁白", "未知", "三人", "大家", "二人", "所有人", "众人", "女孩子",
        "女性", "少女", "姐姐", "妹妹", "哥哥", "弟弟", "朋友A", "朋友B", "朋友C",
    }
    merged = []
    seen = set()
    for source in (segment.get("scene_characters") or [], segment.get("candidates") or []):
        for candidate in source:
            name = str(candidate or "").strip()
            if not name or name in generic or name in seen:
                continue
            seen.add(name)
            merged.append(name)
    return merged[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, help="第五卷 part 序号，如 5（兼容旧模式）")
    ap.add_argument("--snapshot", help="通用模式：包含 segments 的解析快照")
    ap.add_argument("--audit", help="通用模式：包含逐段 reask 的 audit 文件")
    ap.add_argument("--flags", help="通用模式：已有 review_flags 文件")
    ap.add_argument("--roster", help="通用模式：角色清单 JSON")
    ap.add_argument("--output-dir", help="输出目录，默认与 snapshot 同目录")
    ap.add_argument("--output-prefix", help="输出文件前缀，如 muli4_seg8")
    ap.add_argument("--title", help="复核包标题")
    ap.add_argument("--window", type=int, default=2, help="上下文±段数")
    ap.add_argument("--segment-chars", type=int, default=72, help="每段上下文最多保留字符数")
    ap.add_argument("--source-context-chars", type=int, default=0, help="额外附加目标段原文前后文字符数，0 表示关闭")
    args = ap.parse_args()

    if args.snapshot:
        snapshot_path = Path(args.snapshot).expanduser().resolve()
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        pN = snapshot["segments"] if isinstance(snapshot, dict) else snapshot
        out = Path(args.output_dir).expanduser().resolve() if args.output_dir else snapshot_path.parent
        prefix = args.output_prefix or snapshot_path.stem.removesuffix("_parse")
        title = args.title or f"{prefix} 说话人复核包"
        roster_path = Path(args.roster).expanduser().resolve() if args.roster else OUT / "角色清单.json"
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        aliases = _alias_map(roster)
        if args.flags:
            flags = json.loads(Path(args.flags).expanduser().resolve().read_text(encoding="utf-8"))
            items = flags["items"]
        elif args.audit:
            audit = json.loads(Path(args.audit).expanduser().resolve().read_text(encoding="utf-8"))
            audit_segments = audit.get("segments") or {}
            items = []
            for index, segment in enumerate(pN):
                second = audit_segments.get(str(index)) or {}
                suggestion = str(second.get("reask") or "").strip()
                pipeline_speaker = str(segment.get("speaker") or "").strip()
                canonical_suggestion = aliases.get(suggestion, suggestion)
                canonical_pipeline = aliases.get(pipeline_speaker, pipeline_speaker)
                if suggestion and canonical_suggestion != canonical_pipeline:
                    items.append({
                        "segment_index_1based": index + 1,
                        "pipeline_speaker": canonical_pipeline,
                        "singlepass_suggestion": canonical_suggestion,
                    })
        else:
            ap.error("通用模式必须提供 --audit 或 --flags")
    else:
        if not args.part:
            ap.error("请提供 --part，或使用 --snapshot + --audit/--flags 通用模式")
        n = args.part
        off = (n - 1) * 750  # 各 part 750 段（part_006 为 684，但偏移仍按 750 起算）
        src = json.loads(SNAP.read_text(encoding="utf-8"))["segments"]
        pN = src[off:off + 750]
        out = OUT
        prefix = f"muli_5_part_00{n}"
        title = f"第五卷 part_00{n} 说话人复核包"
        roster = json.loads((OUT / "角色清单.json").read_text(encoding="utf-8"))
        flags = json.loads((OUT / f"muli_5_part_00{n}_review_flags.json").read_text(encoding="utf-8"))
        items = flags["items"]

    if not items:
        raise ValueError("没有可生成复核包的分歧项")
    out.mkdir(parents=True, exist_ok=True)
    key = next(k for k in items[0] if k.endswith("index_1based"))

    def ctx(ix: int) -> list[str]:
        i = ix - 1
        rows = []
        for j in range(max(0, i - args.window), min(len(pN), i + args.window + 1)):
            s = pN[j]
            tag = "旁白" if s.get("attribution_type") == "narrator" or s["speaker"] == "旁白" else s["speaker"]
            mark = ">>" if j == i else "  "
            who = "待判" if j == i else tag
            rows.append(f"  {mark}[{j+1}] {who}: {_compact_text(s['text'], args.segment_chars)}")
        return rows

    W = args.window
    head = [
        f"# {title}（{len(items)} 条待判）",
        "",
        "## 角色别名（台词/旁白里出现别名一律归到规范名）",
        _alias_line(roster),
        "",
        "## 判定优先级（从强到弱）",
        "1. 紧邻旁白点名：下一句旁白「X说道/X的声音/X低头/X撂下…」→该台词是 X（注意「…说道：」是引出下一句、不是上句）",
        "2. 称呼≠说话人：台词在喊「X同学/小X/X前辈/宝贝」→说话人**不是** X 本人；「我是X」自我介绍→就是 X",
        "3. 口癖：含「喵」→小柳香穗",
        "4. 优先在【候选/在场】及两方意见中选；原文明示时也可判为旁白或未知临时人物",
        "5. 双人对话严格轮换 A→B→A→B",
        "",
        "## 输出格式（只输出这些行，不要任何解释/复述）",
        "`<编号> <规范名>` = 改判为该角色 ｜ `<编号> K` = 维持流水线 ｜ `<编号> D` = 两可待定",
        "例：`462 小柳香穗`  `029 王冢真唯`  `070 K`  `627 D`",
        "",
        "---",
        "",
    ]
    idmap = {}
    blocks = []
    for it in items:
        ix = it[key]
        target = pN[ix - 1]
        idmap[ix] = target.get("quote_id")
        sc = _review_candidates(target)
        source_context = []
        if args.source_context_chars > 0:
            before = _compact_text(target.get("context_before"), args.source_context_chars)
            after = _compact_text(target.get("context_after"), args.source_context_chars)
            if before:
                source_context.append(f"  原文前文: {before}")
            if after:
                source_context.append(f"  原文后文: {after}")
        blocks.append("\n".join([
            f"### [{ix}]",
            *ctx(ix),
            *source_context,
            f"  候选/在场: {', '.join(sc) if sc else '（未知）'}",
            f"  流水线={it['pipeline_speaker']} ｜ 第二意见={it['singlepass_suggestion']}",
            "",
        ]))

    packet = "\n".join(head) + "\n".join(blocks)
    packet_path = out / f"{prefix}_review_packet.md"
    idmap_path = out / f"{prefix}_review_packet.idmap.json"
    packet_path.write_text(packet, encoding="utf-8")
    idmap_path.write_text(
        json.dumps({str(k): v for k, v in idmap.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {packet_path.name}（{len(items)} 条，上下文±{W}）+ {idmap_path.name}")
    print(f"约 {len(packet)} 字符 — 仅含 flag 紧邻上下文，强模型无需读整卷/原文/跑单遍")


if __name__ == "__main__":
    main()
