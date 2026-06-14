"""script_block.py — 群组聊天/剧本格式分段器（2026-06-14）

某些外传/特殊视角节以「说话人：台词」逐行脚本格式书写（如群组聊天『五女神的房间』），
无「」引号。引号驱动的提取会把整块吞成单个旁白段，丢掉本来**最明确**的说话人信息
（行内显式标注）。本模块把这类旁白段切回逐行段，说话人按别名表归一。

高精度优先：只在一段旁白里出现 ≥3 个"短名+冒号"行内标签、且冒号后不接引号（排除
「某某：「台词」」这类引用式，那类已由引号提取处理）时才触发；普通散文不含重复的
"短名：" 行首，故几乎不会误伤。
"""
from __future__ import annotations

import re
from typing import Callable

from .schema import AttributionType, SegmentEx

# 行内说话人标签：短名（中文/英数/·）后紧跟全/半角冒号
_LABEL_RE = re.compile(r"([一-龥A-Za-z0-9·]{1,8})[：:]")
# 标签左侧应是句界（段首/空白/句末标点），避免把长词的尾巴当成名字
_BOUNDARY = set(" 　\t\r\n。！？!?…」』）)，,、；;　")


def _find_labels(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for m in _LABEL_RE.finditer(text or ""):
        after = text[m.end()] if m.end() < len(text) else ""
        if after in ("「", "『"):
            continue  # 引用式说话标签（已由引号提取处理），跳过
        before = text[m.start() - 1] if m.start() > 0 else ""
        if before == "" or before in _BOUNDARY:
            out.append((m.start(), m.end(), m.group(1)))
    return out


def is_script_block(text: str, min_labels: int = 3) -> bool:
    """该段是否为脚本/群聊格式（≥min_labels 个行内说话人标签）。"""
    labs = _find_labels(text or "")
    if len(labs) < min_labels:
        return False
    names = {n for _, _, n in labs}
    return len(names) >= 2 or len(labs) >= 4


def split_script_block_text(text: str) -> list[tuple[str | None, str]]:
    """切成 [(说话人标签 or None, 台词内容)]；首段无标签的前言 label=None。"""
    labs = _find_labels(text)
    if not labs:
        return [(None, text)]
    out: list[tuple[str | None, str]] = []
    pre = text[: labs[0][0]].strip(" 　\t")
    if pre:
        out.append((None, pre))
    for k, (_s, e, name) in enumerate(labs):
        end = labs[k + 1][0] if k + 1 < len(labs) else len(text)
        content = text[e:end].strip(" 　\t")
        if content:
            out.append((name, content))
    return out


def _is_narration_like(seg: SegmentEx) -> bool:
    sp = str(seg.speaker or "").strip()
    return sp in ("旁白", "未知", "") or seg.attribution_type in (
        AttributionType.NARRATOR, "narrator", AttributionType.UNKNOWN, "unknown",
    )


def apply_script_block_split(
    segments: list[SegmentEx],
    canonicalize: Callable[[str], str] | None = None,
) -> tuple[list[SegmentEx], dict[str, object]]:
    """把脚本/群聊格式的旁白段切成逐行说话人段。返回 (新段列表, stats)。"""
    canon = canonicalize or (lambda x: x)
    stats: dict[str, object] = {"mode": "script_block_split", "blocks": 0, "lines": 0}
    out: list[SegmentEx] = []
    for seg in segments:
        if not _is_narration_like(seg) or not is_script_block(seg.text or ""):
            out.append(seg)
            continue
        pieces = split_script_block_text(seg.text or "")
        # 没有真正切出多条说话人行就放回原段
        if sum(1 for name, _ in pieces if name) < 3:
            out.append(seg)
            continue
        stats["blocks"] = int(stats["blocks"]) + 1
        base = str(seg.quote_id or "seg")
        sub = 0
        for name, content in pieces:
            sub += 1
            qid = f"{base}_s{sub}"
            if name is None:
                out.append(seg.model_copy(update={
                    "quote_id": qid, "speaker": "旁白", "text": content,
                    "attribution_type": AttributionType.NARRATOR, "confidence": 1.0,
                    "evidence": "群组/剧本格式：标签前的前言",
                }))
                continue
            speaker = canon(name) or name
            stats["lines"] = int(stats["lines"]) + 1
            out.append(seg.model_copy(update={
                "quote_id": qid,
                "speaker": speaker,
                "text": content,
                "attribution_type": AttributionType.EXPLICIT_BEFORE,
                "confidence": 0.95,
                "evidence": f"群组/剧本格式：行内显式说话人「{name}」"
                            + (f"→{speaker}" if speaker != name else ""),
                "candidates": [speaker] if speaker else list(seg.candidates or []),
            }))
    return out, stats
