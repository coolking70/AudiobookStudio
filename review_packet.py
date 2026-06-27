"""Offline strong-model review packet helpers.

This module keeps the deterministic parts of the speaker-review workflow out
of the FastAPI endpoints: build a compact Markdown packet, parse a model's
line-based verdicts, and apply those verdicts back to the current segments.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any


GENERIC_SPEAKERS = {
    "旁白",
    "未知",
    "UNKNOWN",
    "三人",
    "大家",
    "二人",
    "所有人",
    "众人",
    "女孩子",
    "女性",
    "少女",
    "姐姐",
    "妹妹",
    "哥哥",
    "弟弟",
    "朋友A",
    "朋友B",
    "朋友C",
}


def _compact_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _normalize_roster(role_hints: Any, segments: list[dict[str, Any]]) -> dict[str, list[str]]:
    roster: dict[str, list[str]] = {}
    if isinstance(role_hints, dict):
        for canon, value in role_hints.items():
            name = str(canon or "").strip()
            if not name:
                continue
            aliases = value if isinstance(value, list) else value.get("aliases", []) if isinstance(value, dict) else []
            roster[name] = [str(alias).strip() for alias in aliases if str(alias or "").strip() and str(alias).strip() != name]
    elif isinstance(role_hints, list):
        for item in role_hints:
            name = str(item or "").strip()
            if name:
                roster.setdefault(name, [])

    for seg in segments:
        names: list[Any] = [seg.get("speaker")]
        names.extend(seg.get("scene_characters") or [])
        names.extend(seg.get("candidates") or [])
        for item in names:
            name = str(item or "").strip()
            if name and name not in GENERIC_SPEAKERS:
                roster.setdefault(name, [])
    return dict(sorted(roster.items(), key=lambda kv: kv[0]))


def _alias_map(roster: dict[str, list[str]]) -> dict[str, str]:
    aliases: dict[str, str] = {"瀬名紫阳花": "濑名紫阳花"}
    for canon, names in roster.items():
        aliases[canon] = canon
        aliases[canon.replace("瀬", "濑")] = canon.replace("瀬", "濑")
        for alias in names:
            aliases[alias] = canon
            aliases[alias.replace("瀬", "濑")] = canon.replace("瀬", "濑")
    return aliases


def canonicalize_name(name: str, aliases: dict[str, str]) -> str:
    value = str(name or "").strip().strip("`，,。；;：:")
    if not value:
        return value
    return aliases.get(value) or aliases.get(value.replace("瀬", "濑")) or value.replace("瀬", "濑")


def _alias_line(roster: dict[str, list[str]]) -> str:
    parts = []
    for canon, names in roster.items():
        clean = [name for name in names if name and name != canon]
        if clean:
            parts.append(f"{canon}←{'/'.join(clean)}")
    return "；".join(parts) or "（当前未提供别名表；请按规范名输出）"


def _review_candidates(segment: dict[str, Any], limit: int = 12) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for source in (segment.get("scene_characters") or [], segment.get("candidates") or []):
        for candidate in source:
            name = str(candidate or "").strip()
            if not name or name in GENERIC_SPEAKERS or name in seen:
                continue
            seen.add(name)
            merged.append(name)
    speaker = str(segment.get("speaker") or "").strip()
    if speaker and speaker not in GENERIC_SPEAKERS and speaker not in seen:
        merged.insert(0, speaker)
    return merged[:limit]


def _extract_second_opinion(segment: dict[str, Any], aliases: dict[str, str]) -> str:
    fields = [
        segment.get("_audit_hint"),
        segment.get("audit_hint"),
        segment.get("_review_hint"),
        segment.get("review_hint"),
        segment.get("evidence"),
        segment.get("_evidence"),
    ]
    text = "；".join(str(item or "") for item in fields if item)
    patterns = [
        r"(?:重问意见|第二意见|异构意见|复核意见|建议(?:改判)?|reask)\s*[=:：]\s*([^\s；;，,。\]\)）]+)",
        r"(?:应为|可能是|建议复核 speaker 为)\s*([^\s；;，,。\]\)）]+)",
        r"([^\s；;，,。\]\)）]+)\s*→\s*([^\s；;，,。\]\)）]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(match.lastindex or 1)
        return canonicalize_name(value, aliases)
    return ""


def _segment_needs_packet(segment: dict[str, Any]) -> bool:
    speaker = str(segment.get("speaker") or "").strip()
    confidence = segment.get("confidence")
    try:
        low_conf = float(confidence) < 0.7
    except (TypeError, ValueError):
        low_conf = False
    return bool(
        segment.get("_audit_tier") == 1
        or segment.get("_needs_review")
        or segment.get("_llm_skipped")
        or segment.get("_skip_reason")
        or speaker in {"未知", "UNKNOWN"}
        or low_conf
    )


def _build_items(segments: list[dict[str, Any]], aliases: dict[str, str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, segment in enumerate(segments):
        if not _segment_needs_packet(segment):
            continue
        pipeline = canonicalize_name(str(segment.get("speaker") or "旁白"), aliases) or "旁白"
        second = _extract_second_opinion(segment, aliases)
        if second and canonicalize_name(second, aliases) == canonicalize_name(pipeline, aliases):
            second = ""
        items.append({
            "segment_index_1based": index + 1,
            "pipeline_speaker": pipeline,
            "singlepass_suggestion": second or "（未给出，仅标记待复核）",
        })
        seen.add(index)

    # If no explicit review markers exist, fall back to the most suspicious
    # machine-readable fields so a freshly imported snapshot still has a packet.
    if not items:
        for index, segment in enumerate(segments):
            if index in seen:
                continue
            evidence = str(segment.get("evidence") or segment.get("_evidence") or "")
            if "复核" not in evidence and "待人工" not in evidence and "冲突" not in evidence:
                continue
            items.append({
                "segment_index_1based": index + 1,
                "pipeline_speaker": canonicalize_name(str(segment.get("speaker") or "旁白"), aliases) or "旁白",
                "singlepass_suggestion": _extract_second_opinion(segment, aliases) or "（未给出，仅标记待复核）",
            })
    return items


def build_review_packet(
    segments: list[dict[str, Any]],
    *,
    source_text: str = "",
    role_hints: Any = None,
    title: str = "说话人复核包",
    window: int = 4,
    segment_chars: int = 120,
    source_context_chars: int = 360,
) -> dict[str, Any]:
    if not segments:
        raise ValueError("当前没有可生成复核包的分析结果。")
    roster = _normalize_roster(role_hints, segments)
    aliases = _alias_map(roster)
    items = _build_items(segments, aliases)
    if not items:
        raise ValueError("当前没有标记为待复核的片段。请先运行机器审计，或在复核工作台标记/补充待复核项。")

    def ctx(ix: int) -> list[str]:
        i = ix - 1
        rows: list[str] = []
        for j in range(max(0, i - window), min(len(segments), i + window + 1)):
            seg = segments[j]
            speaker = str(seg.get("speaker") or "旁白")
            tag = "旁白" if seg.get("attribution_type") == "narrator" or speaker == "旁白" else speaker
            mark = ">>" if j == i else "  "
            who = "待判" if j == i else tag
            rows.append(f"  {mark}[{j + 1}] {who}: {_compact_text(seg.get('text'), segment_chars)}")
        return rows

    head = [
        f"# {title}（{len(items)} 条待判）",
        "",
        "## 使用方式",
        "把本文件完整发送给强模型 AI；让它严格按“输出格式”只返回裁决行。拿到结果后，回到本工具的复核包面板上传/粘贴结果并应用。",
        "",
        "## 角色别名（台词/旁白里出现别名一律归到规范名）",
        _alias_line(roster),
        "",
        "## 判定优先级（从强到弱）",
        "1. 紧邻旁白点名：下一句旁白「X说道/X的声音/X低头/X撂下…」→该台词是 X（注意「…说道：」是引出下一句、不是上句）",
        "2. 称呼≠说话人：台词在喊「X同学/小X/X前辈/宝贝」→说话人不是 X 本人；「我是X」自我介绍→就是 X",
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

    blocks: list[str] = []
    idmap: dict[str, int] = {}
    for item in items:
        ix = int(item["segment_index_1based"])
        target = segments[ix - 1]
        idmap[str(ix)] = ix - 1
        source_context: list[str] = []
        if source_context_chars > 0:
            before = _compact_text(target.get("context_before"), source_context_chars)
            after = _compact_text(target.get("context_after"), source_context_chars)
            if before:
                source_context.append(f"  原文前文: {before}")
            if after:
                source_context.append(f"  原文后文: {after}")
        blocks.append("\n".join([
            f"### [{ix}]",
            *ctx(ix),
            *source_context,
            f"  候选/在场: {', '.join(_review_candidates(target)) or '（未知）'}",
            f"  流水线={item['pipeline_speaker']} ｜ 第二意见={item['singlepass_suggestion']}",
            "",
        ]))

    content = "\n".join(head) + "\n".join(blocks)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", title).strip("_") or "review_packet"
    return {
        "content": content,
        "filename": f"{safe_title}_{stamp}.md",
        "idmap": idmap,
        "items": items,
        "count": len(items),
        "source_text_used": bool(source_text),
    }


def parse_review_verdicts(text: str) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for line in str(text or "").splitlines():
        cleaned = line.strip().strip("`")
        match = re.match(r"^\s*\[?(\d+)\]?\s+(.+?)\s*$", cleaned)
        if not match:
            continue
        value = match.group(2).strip().split()[0].strip("`")
        if value:
            verdicts[str(int(match.group(1)))] = value
    return verdicts


def apply_review_verdicts(
    segments: list[dict[str, Any]],
    verdict_text: str,
    *,
    role_hints: Any = None,
    idmap: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not segments:
        raise ValueError("当前没有可应用复核结果的分析结果。")
    roster = _normalize_roster(role_hints, segments)
    aliases = _alias_map(roster)
    verdicts = parse_review_verdicts(verdict_text)
    if not verdicts:
        raise ValueError("未解析到复核裁决行。请使用格式：<编号> <规范名|K|D>。")

    updated = [dict(seg) for seg in segments]
    applied = kept = deferred = missing = unchanged = 0
    now = datetime.now().isoformat(timespec="seconds")
    for packet_id, raw_value in verdicts.items():
        index = None
        if idmap and packet_id in idmap:
            index = int(idmap[packet_id])
        else:
            index = int(packet_id) - 1
        if index < 0 or index >= len(updated):
            missing += 1
            continue
        seg = updated[index]
        value = raw_value.strip()
        if value.upper() == "K" or value in {"维持", "keep", "KEEP"}:
            seg["_needs_review"] = False
            seg["_manual_reviewed"] = True
            seg["_manual_reviewed_at"] = now
            kept += 1
            continue
        if value.upper() == "D" or value in {"待定", "defer", "DEFER"}:
            seg["_needs_review"] = True
            seg["_review_deferred"] = True
            seg["_manual_reviewed"] = True
            seg["_manual_reviewed_at"] = now
            deferred += 1
            continue
        new_speaker = canonicalize_name(value, aliases)
        old_speaker = str(seg.get("speaker") or "旁白")
        if new_speaker == old_speaker:
            unchanged += 1
        else:
            seg["speaker"] = new_speaker
            seg["attribution_type"] = "narrator" if new_speaker == "旁白" else "manual_review"
            if new_speaker != "旁白":
                candidates = list(seg.get("candidates") or [])
                if new_speaker not in candidates:
                    seg["candidates"] = [new_speaker] + candidates
            evidence = str(seg.get("evidence") or seg.get("_evidence") or "")
            seg["evidence"] = f"离线强模型复核改判：{old_speaker}→{new_speaker}；{evidence}".rstrip("；")
            seg["_evidence"] = seg["evidence"]
            applied += 1
        seg["confidence"] = max(float(seg.get("confidence") or 0.0), 0.9)
        seg["_confidence"] = "high"
        seg["_needs_review"] = False
        seg["_review_deferred"] = False
        seg["_manual_reviewed"] = True
        seg["_manual_reviewed_at"] = now

    return {
        "segments": updated,
        "summary": {
            "parsed": len(verdicts),
            "applied": applied,
            "kept": kept,
            "deferred": deferred,
            "unchanged": unchanged,
            "missing": missing,
        },
    }


def encode_packet_state(packet: dict[str, Any]) -> str:
    return json.dumps({"idmap": packet.get("idmap") or {}}, ensure_ascii=False)
