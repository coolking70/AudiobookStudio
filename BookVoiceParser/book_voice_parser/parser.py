from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from .alias_registry import AliasRegistry, clean_role_hint_name
from .address_term_backcheck import apply_address_term_backcheck
from .block_review import apply_block_review
from .candidate_gen import generate_candidates
from .cleaner import normalize_text
from .consistency_fixer import fix_consistency
from .implicit_attributor import attribute_implicit
from .nlp_backend import HanLPNERBackend, NERBackend
from .quote_extractor import extract_quotes
from .review_router import collect_review_items
from .rule_attributor import attribute_explicit
from .schema import Attribution, AttributionType, CandidateSet, ParseResult, QuoteSpan, SegmentEx
from .spc_ranker import CandidateRanker, OpenAICompatibleSPCRanker


NARRATOR_CUE_ONLY_RE = re.compile(
    r"^[，,。；;：:\s]*[一-龥A-Za-z0-9_]{1,24}"
    r"(?:对|向|冲)?[一-龥A-Za-z0-9_]{0,12}"
    r"(?:喃喃道|嘀咕道|低声道|轻声道|笑道|冷笑道|喝道|骂道|叹道|缓缓开口|开口|心想|想道|说道|说|问道|问|答道|答|喊道|喊|叫道|叫|道)"
    r"[：:，,。\s]*$"
)

# P1b：匹配「我」但排除「我们/我方/我国/我家/我辈/我等」
_FIRST_PERSON_RE = re.compile(r"我(?!们|方|国|家|辈|等)")

# P1c：发送消息上下文检测
# 匹配「我…回/发了（一条/一封）讯息/消息/短信」模式（允许中间有任意非句末内容）。
# 允许发送动词和消息名词之间有量词（如"一条"），例：「我发了一条消息给妈妈」。
_MSG_SEND_RE = re.compile(
    r"我[^。！？\n]{0,30}?(?:回复|发送|转发|回|发|传|送)了[^。！？\n]{0,8}?(?:讯息|消息|短信|信息)",
    re.DOTALL,
)

# 章节标题识别（仅识别结构性章节，不做激进分割）
# 使用 ^\s* 允许行首有空格（部分电子书格式每行有前置空白）
_CHAPTER_HEADING_RE = re.compile(
    r"(?m)^\s*("
    r"第[零一二三四五六七八九十百千万\d]+[章节回卷]"
    r"|序章|终章|尾声|幕间|插话|后记|番外|特典"
    r"|第[零一二三四五六七八九十百千万\d]+卷\s*特典.*"
    r"|Chapter\s+\d+"
    r").*$"
)


# 纯标点/连词微段（相邻引号之间的连接字符，如「嗯」和「这样啊」中的"和"）
_MICRO_CONNECTOR_RE = re.compile(r"^[，。、！？…—和与或及以]+$")


def _detect_chapter_offsets(cleaned: str) -> list[int]:
    """返回 cleaned 文本中各章节标题起始字符偏移（升序）。"""
    return [m.start() for m in _CHAPTER_HEADING_RE.finditer(cleaned)]


def _chapter_opening(cleaned: str, chapter_offset: int, next_offset: int | None = None, window: int = 400) -> str:
    """返回章节标题之后的开头文本（用于视角检测），不越过下一章节边界。"""
    start = cleaned.find("\n", chapter_offset)
    if start == -1:
        start = chapter_offset
    end = min(start + window, next_offset if next_offset is not None else len(cleaned))
    return cleaned[start:end]


def _heading_has_role_narrator_pattern(heading: str, alias: str) -> bool:
    return any(
        f"{alias}{suffix}" in heading
        for suffix in ("的故事", "的物語", "的音档", "的音檔", "の物語")
    )


def _extract_chapter_narrator(heading: str, role_hints: list[str]) -> str | None:
    """
    从「X的故事」「X的物語」章节标题中提取该章节的叙述者角色名（规范化全名）。
    支持全名与后 2/3 字简称匹配。找不到时返回 None。
    """
    for role in role_hints:
        aliases_to_check: list[str] = [role]
        if len(role) >= 3:
            aliases_to_check.append(role[-2:])
        if len(role) >= 4:
            aliases_to_check.append(role[-3:])
        for alias in aliases_to_check:
            if _heading_has_role_narrator_pattern(heading, alias):
                return role
    return None


def _role_title_aliases(role: str) -> set[str]:
    role = str(role or "").strip()
    if not role:
        return set()
    aliases = {role}
    chinese = "".join(re.findall(r"[一-龥]+", role))
    if len(chinese) >= 2:
        aliases.add(chinese[-2:])
        aliases.add("小" + chinese[-2:])
    if len(chinese) >= 3:
        aliases.add(chinese[-3:])
        aliases.add("小" + chinese[-3:])
        aliases.add("小" + chinese[-3:-1])
    return {item for item in aliases if len(item) >= 2}


def _conservative_title_aliases(role: str) -> set[str]:
    role = str(role or "").strip()
    if not role:
        return set()
    aliases = {role}
    chinese = "".join(re.findall(r"[一-龥]+", role))
    if len(chinese) >= 4:
        aliases.add(chinese[-3:])
    if len(chinese) >= 2:
        aliases.add("小" + chinese[-2:])
    if len(chinese) >= 3:
        aliases.add("小" + chinese[-3:])
    ambiguous = {"恋人", "朋友", "同学", "小姐", "姐姐", "妹妹", "妈妈", "母亲", "主人", "宠物"}
    return {item for item in aliases if len(item) >= 3 and item not in ambiguous}


def _match_audio_track_narrator(heading: str, role_hints: list[str]) -> str | None:
    heading = str(heading or "")
    if not heading or ("的音档" not in heading and "的音檔" not in heading):
        return None
    for role in role_hints:
        for alias in _role_title_aliases(role):
            if f"{alias}的音档" in heading or f"{alias}的音檔" in heading:
                return role
    return None


def _match_title_named_role(heading: str, role_hints: list[str]) -> str | None:
    heading = str(heading or "")
    if not heading:
        return None
    direct = _match_audio_track_narrator(heading, role_hints)
    if direct:
        return direct
    matched: list[tuple[int, str]] = []
    for role in role_hints:
        for alias in _conservative_title_aliases(role):
            if alias in heading:
                matched.append((len(alias), role))
                break
    if len({role for _, role in matched}) != 1:
        return None
    return matched[0][1]


def _extract_dual_chapter_roles(heading: str, role_hints: list[str]) -> list[str]:
    """Return canonical roles from titles like 「遥奈与紫阳花的故事」."""
    dual_m = re.search(
        r"([一-龥A-Za-z0-9]{2,8})与([一-龥A-Za-z0-9]{2,8})(?:的故事|的物語)", heading
    )
    if not dual_m:
        return []
    matched_roles: list[str] = []
    for name in (dual_m.group(1), dual_m.group(2)):
        for role in role_hints:
            if role == name or role.endswith(name) or role.startswith(name):
                if role not in matched_roles:
                    matched_roles.append(role)
                break
    return matched_roles


# 非叙事性元段落关键词：这类章节没有小说内视角，跳过 LLM 叙述者检测
_META_SECTION_KEYWORDS = frozenset(["后记", "後記", "插图", "插圖", "前言", "附录", "附錄"])


def _extract_chapter_narrator_hint(heading: str, role_hints: list[str]) -> str | None:
    """
    生成供 LLM 参考的章节叙述者提示文字（比 _extract_chapter_narrator 更宽泛）：
    - 「X的故事」→ X 的规范全名
    - 「X与Y的故事」→ "X、Y（双角色章节）"
    找不到模式时返回 None。
    """
    # 双角色模式：X与Y的故事 / X与Y的物語
    matched_roles = _extract_dual_chapter_roles(heading, role_hints)
    if matched_roles:
        return "、".join(matched_roles) + "（双角色章节）"

    # 单角色模式
    single = _extract_chapter_narrator(heading, role_hints)
    return single


def _title_fallback_narrator(heading: str, main_narrator: str, role_hints: list[str]) -> str | None:
    """Conservative fallback used only after the shift detector says "shift"."""
    dual_roles = _extract_dual_chapter_roles(heading, role_hints)
    if dual_roles and dual_roles[0] != main_narrator:
        return dual_roles[0]
    single = _extract_chapter_narrator(heading, role_hints)
    if single and single != main_narrator:
        return single
    return None


def _is_perspective_shift_chapter(
    chapter_heading: str,
    chapter_opening_text: str,
    narrator: str,
    role_hints: list[str] | None = None,
) -> bool:
    """
    启发式检测视角转换章节，两种信号任一成立即返回 True：

    信号1：叙述者姓名出现在引号外叙述文字中（叙述者被第三方描述）。
    信号2：章节标题包含「[他人角色名]的故事」等模式（明确他者视角章节）。
    """
    if not narrator or len(narrator) < 2:
        return False

    # 信号1：叙述者名在非引号叙述中出现（排除「我是X」「X就是我」等自称表达）
    stripped = re.sub(r"「[^」]*」", "", chapter_opening_text)
    narrator_short = narrator[-2:] if len(narrator) >= 2 else narrator
    if narrator in stripped or narrator_short in stripped:
        # 自称模式：叙述者用自己名字自我介绍，不视为视角转换
        # 同时检查全名和简称，且允许全名在「我是」后间隔任意长度出现
        _self_ref = re.compile(
            r"我[是叫就为—]{0,2}" + re.escape(narrator)
            + r"|我[是叫就为—]{0,2}" + re.escape(narrator_short)
            + r"|" + re.escape(narrator) + r"[就是—]{0,2}我"
            + r"|" + re.escape(narrator_short) + r"[就是—]{0,2}我"
        )
        if not _self_ref.search(stripped):
            return True

    # 信号2：标题含「[非叙述者角色]的故事/物語」
    if role_hints and chapter_heading:
        for role in role_hints:
            if role == narrator:
                continue
            aliases = [role]
            if len(role) >= 3:
                aliases.append(role[-2:])
            if len(role) >= 4:
                aliases.append(role[-3:])
            for alias in aliases:
                if _heading_has_role_narrator_pattern(chapter_heading, alias):
                    return True

    return False


def _clean_narrator_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = cleaned.strip(" \t\r\n")
    if not cleaned:
        return ""
    # 过滤相邻引号间的单字/双字连词或纯标点（TTS 无法处理）
    if len(cleaned) <= 2 and _MICRO_CONNECTOR_RE.fullmatch(cleaned):
        return ""
    if NARRATOR_CUE_ONLY_RE.fullmatch(cleaned):
        return ""
    parts = re.split(r"([。！？!?])", cleaned)
    rebuilt: list[str] = []
    for idx in range(0, len(parts), 2):
        sentence = (parts[idx] or "").strip()
        punct = parts[idx + 1] if idx + 1 < len(parts) else ""
        if not sentence:
            continue
        candidate = f"{sentence}{punct}"
        if NARRATOR_CUE_ONLY_RE.fullmatch(candidate):
            continue
        rebuilt.append(candidate)
    if rebuilt:
        cleaned = " ".join(rebuilt).strip()
    elif NARRATOR_CUE_ONLY_RE.search(cleaned):
        return ""
    return cleaned


_AUDIO_TRACK_SOURCE = "audio_track_narrator"
_SPECIAL_PERSPECTIVE_SOURCE = "special_perspective_chapter"
_SCENE_ACTIVE_SOURCE = "scene_active"
_SCENE_MENTION_SOURCE = "scene_semantic_mention"

_SCENE_ENTRY_RE = re.compile(
    r"(?:出现|走来|走进|进来|回来|来到|靠近|坐下|站在|搭话|开口|说道|说|问道|问|回答|喊|叫|挥手|看着|望着|注视|笑)"
)
_SCENE_EXIT_RE = re.compile(r"(?:离开|走了|走远|退场|不见|消失|回去|出门|离席)")
_SCENE_RESET_RE = re.compile(r"(?:\*{3,}|——+|-{3,}|场景切换|时间来到|隔天|第二天|翌日)")


def _is_audio_track_stop_text(text: str) -> bool:
    text = str(text or "").strip()
    return bool(text and ("用户使用感评价" in text or "使用感评价" in text))


def _is_audio_track_meta_text(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return True
    if re.search(r"track\s*\d+\.MP3", text, flags=re.IGNORECASE):
        return True
    if "的音档" in text or "的音檔" in text:
        return True
    return False


def _is_parenthetical_sound_effect(text: str) -> bool:
    stripped = str(text or "").strip()
    if not (stripped.startswith(("（", "(")) and stripped.endswith(("）", ")"))):
        return False
    inner = stripped[1:-1]
    return any(token in inner for token in ("声音", "声", "音效", "开门", "关门", "脚步", "铃声", "敲门"))


def _is_parenthetical_inner_voice(text: str) -> bool:
    stripped = str(text or "").strip()
    return bool(stripped.startswith(("（", "(")) and stripped.endswith(("）", ")")) and not _is_parenthetical_sound_effect(stripped))


def _text_has_address_to_other(text: str, speaker: str, role_hints: list[str]) -> bool:
    text = str(text or "")
    if not text:
        return False
    speaker_aliases = _role_title_aliases(speaker)
    for role in role_hints:
        if role == speaker:
            continue
        for alias in _role_title_aliases(role):
            if alias in text and alias not in speaker_aliases:
                return True
    return any(suffix in text for suffix in ("亲", "小姐", "同学", "姐姐", "妹妹", "哥哥", "弟弟", "さん", "ちゃん"))


def _is_speech_like_special_segment(seg: SegmentEx) -> bool:
    speaker = str(seg.speaker or "").strip()
    if speaker not in {"", "旁白", "未知", "UNKNOWN"}:
        return True
    return _is_parenthetical_inner_voice(seg.text)


def _special_chapter_profile(
    section: list[SegmentEx],
    speaker: str,
    role_hints: list[str],
) -> dict[str, object]:
    content = [
        seg for seg in section
        if not _is_audio_track_stop_text(seg.text)
        and not _is_audio_track_meta_text(seg.text)
        and not _is_parenthetical_sound_effect(seg.text)
    ]
    if not content:
        return {
            "content_count": 0,
            "speech_like_ratio": 0.0,
            "address_count": 0,
            "address_ratio": 0.0,
            "narrator_context_ratio": 0.0,
            "is_special": False,
        }
    speech_like = sum(1 for seg in content if _is_speech_like_special_segment(seg))
    address_count = sum(1 for seg in content if _text_has_address_to_other(seg.text, speaker, role_hints))
    narrator_context = sum(
        1 for seg in content
        if str(seg.speaker or "").strip() == "旁白" and not _is_parenthetical_inner_voice(seg.text)
    )
    content_count = len(content)
    speech_ratio = speech_like / content_count
    address_ratio = address_count / content_count
    narrator_ratio = narrator_context / content_count
    is_special = (
        content_count >= 6
        and speech_ratio >= 0.70
        and narrator_ratio <= 0.30
        and (address_count >= 3 or address_ratio >= 0.35)
    )
    return {
        "content_count": content_count,
        "speech_like_ratio": round(speech_ratio, 3),
        "address_count": address_count,
        "address_ratio": round(address_ratio, 3),
        "narrator_context_ratio": round(narrator_ratio, 3),
        "is_special": is_special,
    }


def _lock_special_perspective_segment(seg: SegmentEx, speaker: str, profile: dict[str, object]) -> None:
    old = str(seg.speaker or "").strip()
    seg.speaker = speaker
    seg.confidence = max(float(seg.confidence or 0.0), 0.95)
    seg.attribution_type = AttributionType.IMPLICIT
    if speaker not in (seg.candidates or []):
        seg.candidates = [speaker, *list(seg.candidates or [])]
    if speaker not in (seg.scene_characters or []):
        seg.scene_characters = [speaker, *list(seg.scene_characters or [])]
    sources = dict(seg.candidate_sources or {})
    speaker_sources = list(sources.get(speaker) or [])
    for source in (_SPECIAL_PERSPECTIVE_SOURCE, _AUDIO_TRACK_SOURCE):
        if source not in speaker_sources:
            speaker_sources.append(source)
    sources[speaker] = speaker_sources
    seg.candidate_sources = sources
    note = (
        f"特殊视角章节：标题角色 {speaker}，"
        f"连续对白/称呼特征 content={profile.get('content_count')}, "
        f"speech={profile.get('speech_like_ratio')}, address={profile.get('address_count')}"
    )
    if old and old != speaker:
        note = f"{note}，由 {old} 修正为 {speaker}"
    evidence = str(seg.evidence or "")
    if note not in evidence:
        seg.evidence = f"{evidence}；{note}" if evidence else note


def _apply_special_perspective_chapter_overrides(
    segments: list[SegmentEx],
    role_hints: list[str],
) -> tuple[list[SegmentEx], dict[str, object]]:
    updated = [seg.model_copy() for seg in segments]
    stats: dict[str, object] = {
        "mode": "special_perspective_chapter",
        "sections": 0,
        "detected": 0,
        "locked": 0,
        "skipped_meta": 0,
        "stopped": 0,
        "rejected": 0,
        "profiles": [],
    }
    if not role_hints:
        return updated, stats

    heading_indexes = [
        idx for idx, seg in enumerate(updated)
        if "章节标题" in str(seg.evidence or "") or "chapter_heading" in str(seg.candidate_sources or {})
    ]
    if not heading_indexes:
        return updated, stats

    for order, heading_index in enumerate(heading_indexes):
        heading = updated[heading_index]
        speaker = _match_title_named_role(heading.text, role_hints)
        if not speaker:
            continue
        next_heading = heading_indexes[order + 1] if order + 1 < len(heading_indexes) else len(updated)
        stop_index = next_heading
        for idx in range(heading_index + 1, next_heading):
            if _is_audio_track_stop_text(updated[idx].text):
                stop_index = idx
                stats["stopped"] = int(stats["stopped"]) + 1
                break
        section = updated[heading_index + 1: stop_index]
        profile = _special_chapter_profile(section, speaker, role_hints)
        profile_record = {
            "heading": str(heading.text or "")[:80],
            "speaker": speaker,
            **profile,
        }
        stats["profiles"] = [*list(stats["profiles"]), profile_record]
        stats["sections"] = int(stats["sections"]) + 1
        if not profile.get("is_special"):
            stats["rejected"] = int(stats["rejected"]) + 1
            continue
        stats["detected"] = int(stats["detected"]) + 1
        for seg in section:
            if _is_audio_track_stop_text(seg.text):
                break
            if _is_audio_track_meta_text(seg.text) or _is_parenthetical_sound_effect(seg.text):
                stats["skipped_meta"] = int(stats["skipped_meta"]) + 1
                continue
            _lock_special_perspective_segment(seg, speaker, profile)
            stats["locked"] = int(stats["locked"]) + 1

    return updated, stats


def _append_segment_evidence(seg: SegmentEx, note: str) -> None:
    if not note:
        return
    evidence = str(seg.evidence or "")
    if note in evidence:
        return
    seg.evidence = f"{evidence}；{note}" if evidence else note


def _add_segment_candidate_source(seg: SegmentEx, speaker: str, source: str) -> None:
    speaker = str(speaker or "").strip()
    if not speaker or speaker in {"旁白", "未知", "UNKNOWN"}:
        return
    if speaker not in (seg.candidates or []):
        seg.candidates = [speaker, *list(seg.candidates or [])]
    if speaker not in (seg.scene_characters or []):
        seg.scene_characters = [speaker, *list(seg.scene_characters or [])]
    sources = dict(seg.candidate_sources or {})
    values = list(sources.get(speaker) or [])
    if source not in values:
        values.append(source)
    sources[speaker] = values
    seg.candidate_sources = sources


# 通用称谓/头衔本身不指向具体角色，不能作为"某角色登场"的线索。
# 否则 "平野同学" 会生成裸别名 "同学"，匹配文中所有 "X同学" 而被误判为在场。
_GENERIC_TITLE_TERMS = {
    "同学", "老师", "前辈", "学长", "学姐", "学妹", "学弟", "同志",
    "小姐", "先生", "女士", "大人", "殿下", "陛下",
    "恋人", "朋友", "主人", "宠物", "队长", "老板", "经理",
    "姐姐", "妹妹", "哥哥", "弟弟", "妈妈", "母亲", "爸爸", "父亲",
    "阿姨", "叔叔", "大叔", "大姐", "大哥",
}


def _is_generic_scene_alias(alias: str) -> bool:
    """裸通用称谓，或 '单字 + 通用称谓' 这类过宽的派生别名（如 同学/小同学/野同学）。"""
    if alias in _GENERIC_TITLE_TERMS:
        return True
    return len(alias) <= 3 and any(
        alias != term and alias.endswith(term) for term in _GENERIC_TITLE_TERMS
    )


def _role_aliases_for_scene(role: str) -> set[str]:
    aliases = _role_title_aliases(role)
    aliases.update(_conservative_title_aliases(role))
    aliases.discard(role[-1:] if role else "")
    return {
        alias for alias in aliases
        if len(alias) >= 2 and not _is_generic_scene_alias(alias)
    }


def _roles_with_action_context(text: str, role_hints: list[str], action_re: re.Pattern[str]) -> set[str]:
    text = str(text or "")
    found: set[str] = set()
    if not text:
        return found
    for role in role_hints:
        for alias in _role_aliases_for_scene(role):
            start = 0
            while True:
                idx = text.find(alias, start)
                if idx < 0:
                    break
                window = text[max(0, idx - 24): idx + len(alias) + 36]
                if action_re.search(window):
                    found.add(role)
                    break
                start = idx + len(alias)
    return found


def _is_scene_boundary_segment(seg: SegmentEx) -> bool:
    evidence = str(seg.evidence or "")
    text = str(seg.text or "")
    return "章节标题" in evidence or "chapter_heading" in str(seg.candidate_sources or {}) or bool(_SCENE_RESET_RE.search(text))


def _speaker_is_scene_candidate(speaker: str) -> bool:
    return bool(speaker and speaker not in {"旁白", "未知", "UNKNOWN"})


def _has_scene_strong_source(seg: SegmentEx, speaker: str) -> bool:
    sources = set((seg.candidate_sources or {}).get(speaker, []) or [])
    strong = {
        "role_hints",
        "title",
        "rule_cue",
        "group_cue",
        "appearance_alias",
        "hanlp_ner",
        "relation_conditional",
        "address_term_backcheck",
        "address_term_local_context",
        _SPECIAL_PERSPECTIVE_SOURCE,
        _SCENE_ACTIVE_SOURCE,
    }
    return bool(sources & strong)


def _apply_scene_state_constraints(
    segments: list[SegmentEx],
    role_hints: list[str],
    narrator: str | None = None,
    review_threshold: float = 0.7,
    aliases: AliasRegistry | None = None,
) -> tuple[list[SegmentEx], dict[str, object]]:
    updated = [seg.model_copy() for seg in segments]
    active: list[str] = []
    recent_confirmed: list[str] = []
    stats: dict[str, object] = {
        "mode": "scene_state_soft_constraints",
        "boundaries": 0,
        "entered": 0,
        "left": 0,
        "active_hits": 0,
        "scene_sources_added": 0,
        "out_of_scene_marked": 0,
        "confidence_boosted": 0,
    }

    def canonical_role(name: str | None) -> str:
        value = str(name or "").strip()
        return aliases.canonicalize(value) if aliases is not None else value

    def reset_active(seed: list[str] | None = None) -> None:
        active.clear()
        narrator_name = canonical_role(narrator)
        if narrator_name:
            active.append(narrator_name)
        for name in seed or []:
            canonical = canonical_role(name)
            if _speaker_is_scene_candidate(canonical) and canonical not in active:
                active.append(canonical)

    def add_active(name: str, reason: str, seg: SegmentEx | None = None) -> None:
        name = canonical_role(name)
        if not _speaker_is_scene_candidate(name):
            return
        if name not in active:
            active.append(name)
            stats["entered"] = int(stats["entered"]) + 1
        if seg is not None:
            _add_segment_candidate_source(seg, name, _SCENE_MENTION_SOURCE)
            _append_segment_evidence(seg, f"场景状态：{name} {reason}")

    def remove_active(name: str) -> None:
        canonical = canonical_role(name)
        if canonical in active and canonical != canonical_role(narrator):
            active.remove(canonical)
            stats["left"] = int(stats["left"]) + 1

    reset_active()
    for seg in updated:
        text = str(seg.text or "")
        speaker = canonical_role(seg.speaker)

        if _is_scene_boundary_segment(seg):
            seed: list[str] = []
            title_role = _match_title_named_role(text, role_hints)
            if title_role:
                seed.append(canonical_role(title_role))
            stats["boundaries"] = int(stats["boundaries"]) + 1
            reset_active(seed)
            continue

        # 旁白和上下文中的明确动作/登场/离场才改变活跃集合；台词正文里的名字默认视作受话人或被提及者。
        semantic_text = "\n".join([
            str(seg.context_before or "")[-180:],
            text if speaker == "旁白" else "",
        ])
        for role in _roles_with_action_context(semantic_text, role_hints, _SCENE_ENTRY_RE):
            add_active(role, "由动作/登场语义加入当前场景", seg)
        for role in _roles_with_action_context(semantic_text, role_hints, _SCENE_EXIT_RE):
            remove_active(role)

        confidence = float(seg.confidence or 0.0)
        special_locked = _SPECIAL_PERSPECTIVE_SOURCE in set((seg.candidate_sources or {}).get(speaker, []) or [])

        if _speaker_is_scene_candidate(speaker):
            if speaker in active:
                _add_segment_candidate_source(seg, speaker, _SCENE_ACTIVE_SOURCE)
                stats["scene_sources_added"] = int(stats["scene_sources_added"]) + 1
                stats["active_hits"] = int(stats["active_hits"]) + 1
                if review_threshold <= confidence < 0.85 and "LLM复核待人工" not in str(seg.evidence or ""):
                    seg.confidence = min(0.85, confidence + 0.06)
                    stats["confidence_boosted"] = int(stats["confidence_boosted"]) + 1
            elif confidence >= 0.85 or special_locked or _has_scene_strong_source(seg, speaker):
                add_active(speaker, "由高置信说话人加入当前场景", seg)
                _add_segment_candidate_source(seg, speaker, _SCENE_ACTIVE_SOURCE)
                stats["scene_sources_added"] = int(stats["scene_sources_added"]) + 1
            elif active and speaker not in recent_confirmed[-3:]:
                seg.confidence = min(confidence, max(0.55, review_threshold - 0.01))
                _append_segment_evidence(seg, f"场景状态：{speaker} 不在当前活跃候选 {active[:6]}，作为软约束降置信")
                stats["out_of_scene_marked"] = int(stats["out_of_scene_marked"]) + 1

            if seg.confidence >= review_threshold and speaker not in {"旁白", "未知", "UNKNOWN"}:
                recent_confirmed.append(speaker)
                if len(recent_confirmed) > 12:
                    recent_confirmed[:] = recent_confirmed[-12:]

    return updated, stats


def _is_dialogue_fragment(text: str) -> bool:
    """
    检测旁白区的文本段是否实为漏提取的引号或被段落分隔符切断的对话碎片。

    触发条件（二选一）：
    1. 整体被单对 「」 包裹（= 漏提取的引号，通常含嵌套引号或其他边缘情况）
    2. 「/」 括号不平衡（= 某段对话被 \\n\\n 切成了头尾两截）

    注意：不包含引号的纯叙述文本不受影响。含有嵌入引用（如「词语」了一声）
    的叙述文本因括号平衡也不受影响。
    """
    open_count = text.count("「")
    close_count = text.count("」")
    if open_count == 0 and close_count == 0:
        return False
    # 整体被单对 「」 包裹
    if text.startswith("「") and text.endswith("」") and open_count == 1 and close_count == 1:
        return True
    # 括号不平衡（对话碎片）
    return open_count != close_count


def _append_narrator_segments(
    segments: list[SegmentEx],
    raw_text: str,
    quote_id_prefix: str,
    context_chars: int = 120,
) -> None:
    cleaned = _clean_narrator_text(raw_text)
    if not cleaned:
        return
    # 保持段落感，避免把整章旁白塞成单个超长 TTS 片段。
    parts = [part.strip() for part in re.split(r"\n\s*\n+", cleaned) if part.strip()] or [cleaned]
    for idx, part in enumerate(parts, start=1):
        if _is_dialogue_fragment(part):
            # 括号不平衡或整体被单对「」包裹 → 疑似漏提取的对话，标记为待复核
            segments.append(
                SegmentEx(
                    quote_id=f"{quote_id_prefix}_n{idx}",
                    speaker="未知",
                    text=part,
                    confidence=0.25,
                    evidence="旁白区含对话括号，疑似漏提取引号",
                    attribution_type=AttributionType.UNKNOWN,
                    candidates=["旁白", "未知"],
                    candidate_sources={"旁白": ["narrator_gap"], "未知": ["dialogue_fragment"]},
                    scene_characters=[],
                    context_before=part[:context_chars],
                    context_after=part[-context_chars:],
                )
            )
        else:
            segments.append(
                SegmentEx(
                    quote_id=f"{quote_id_prefix}_n{idx}",
                    speaker="旁白",
                    text=part,
                    confidence=1.0,
                    evidence="引号外叙述文本",
                    attribution_type=AttributionType.NARRATOR,
                    candidates=["旁白"],
                    candidate_sources={"旁白": ["narrator_gap"]},
                    scene_characters=[],
                    context_before=part[:context_chars],
                    context_after=part[-context_chars:],
                )
            )


# ── 辅助：解包外层 role_hints 包装结构 ──────────────────────────────────────

def _unwrap_role_hints(
    role_hints: Any,
    narrator: str | None = None,
) -> tuple[Any, str | None]:
    """
    解包外部 AI 角色提取结果常用的 {"narrator": "...", "roles": {...}} 包装格式。

    若 role_hints 是含 "roles" 键的字典，则提取内层 roles 字典作为实际 role_hints；
    同时若 narrator 未设置，从 "narrator" 键中提取叙述者名。

    Returns:
        (unwrapped_role_hints, resolved_narrator)
    """
    if not isinstance(role_hints, dict):
        return role_hints, narrator
    if "roles" in role_hints and isinstance(role_hints["roles"], dict):
        inner = role_hints["roles"]
        resolved_narrator = narrator
        if resolved_narrator is None and isinstance(role_hints.get("narrator"), str):
            resolved_narrator = role_hints["narrator"].strip() or None
        return inner, resolved_narrator
    return role_hints, narrator


# ── 辅助：规范化 role_hints 为扁平列表 ───────────────────────────────────────

def _normalize_role_hints(
    role_hints: dict[str, list[str] | str] | list[str] | None,
) -> list[str]:
    """将各种 role_hints 格式归一为角色名字符串列表。"""
    if not role_hints:
        return []
    if isinstance(role_hints, list):
        return [name for item in role_hints if (name := clean_role_hint_name(item))]
    if isinstance(role_hints, dict):
        return [name for key in role_hints.keys() if (name := clean_role_hint_name(key))]
    return []


_TEMPORARY_SCENE_SOURCE = "temporary_scene"


def _candidate_sources_for(cset, speaker: str) -> set[str]:
    if not cset or not speaker:
        return set()
    return {str(item) for item in (cset.candidate_sources or {}).get(speaker, []) if item}


def _has_temporary_scene_candidate(cset) -> bool:
    if not cset:
        return False
    for name in cset.candidates or []:
        if _TEMPORARY_SCENE_SOURCE in _candidate_sources_for(cset, name):
            return True
    return False


def _guard_batch_llm_attribution(attr: Attribution, cset) -> None:
    """Demote narrator-only guesses when unnamed people are present."""
    if not cset or not attr or attr.speaker in {"旁白", "未知", "UNKNOWN"}:
        return
    sources = _candidate_sources_for(cset, attr.speaker)
    if sources <= {"narrator_anchor"} and _has_temporary_scene_candidate(cset):
        if attr.confidence > 0.69:
            attr.confidence = 0.69
        note = "仅叙述者锚点支持，场景含未命名人物"
        attr.evidence = f"{attr.evidence}；{note}" if attr.evidence else note


# ── P1a：自我介绍检测 ────────────────────────────────────────────────────────

def _attribute_self_identified(
    quote: QuoteSpan,
    role_hints: list[str],
) -> Attribution | None:
    """
    检测台词文本中是否含自我介绍模式：
      「我是王冢真唯」「我王冢真唯...」「我叫真唯」等

    仅匹配 role_hints 中的已知角色（全名或后 2/3 字简称）。
    返回 confidence=0.95 的显性归因，不对未知名字猜测。
    """
    if not role_hints:
        return None
    text = quote.text or ""
    if "我" not in text:
        return None

    for role in role_hints:
        aliases_to_check: list[tuple[str, float]] = [(role, 0.95)]  # (alias, confidence)
        if len(role) >= 3:
            aliases_to_check.append((role[-2:], 0.90))
        if len(role) >= 4:
            aliases_to_check.append((role[-3:], 0.90))

        for alias, conf in aliases_to_check:
            for connector in ["", "是", "叫", "乃", "就是"]:
                pattern = f"我{connector}{alias}"
                if pattern in text:
                    return Attribution(
                        quote_id=quote.quote_id,
                        speaker=role,
                        confidence=conf,
                        evidence=f"自我介绍：「{pattern[:18]}」",
                        attribution_type=AttributionType.EXPLICIT_BEFORE,
                    )
    return None


# ── P1b：一人称叙述者锚点 ────────────────────────────────────────────────────

# 长辈称呼守卫：若台词含「姐姐/哥哥」等向年长者的称呼，说话人是在称呼叙述者，而非叙述者自身。
# 注：只拦截对上位亲属的称呼（「姐姐/哥哥」），不拦叙述者对下位亲属的称呼（「妹妹/弟弟」）。
# 注意：[大人]? 是字符类，需用 (?:大人)? 匹配两字字符串。
_ELDER_VOCATIVE_RE = re.compile(
    r"(?:^|[，,\s「])(?:姐姐|哥哥)(?:大人)?(?=[，,\s!！?？」]|$)"
)


def _attribute_narrator_anchored(
    quote: QuoteSpan,
    narrator: str,
) -> Attribution | None:
    """
    台词含「我」时，归因给主视角叙述者（日式轻小说常用主角固定视角）。

    置信度 0.82：中高置信，允许 fix_consistency 层进一步调整。

    误触发守卫：
    1. 叙述者简称出现在台词内 → 叙述者是受话对象，非说话人
       例：「怎么了，玲奈子？看来，是重新迷上我了吗？」→ 真唯在对玲奈子说话。
    2. 台词含长辈称呼「姐姐/哥哥」→ 说话人是在称呼叙述者（年幼角色），而非叙述者自身
       例：「嘿嘿，姐姐大人，我的炸鸡分你一块吧？」→ 妹妹在对叙述者说话。
    """
    if not _FIRST_PERSON_RE.search(quote.text or ""):
        return None
    text = quote.text or ""
    # 守卫1：叙述者名字（全名或末2字简称）出现在台词内 → 是受话对象，非说话人
    narrator_short = narrator[-2:] if len(narrator) >= 2 else narrator
    if narrator in text or (narrator_short and narrator_short in text):
        return None
    # 守卫2：台词含长辈称呼 → 说话人比叙述者年幼，不是叙述者
    if _ELDER_VOCATIVE_RE.search(text):
        return None
    return Attribution(
        quote_id=quote.quote_id,
        speaker=narrator,
        confidence=0.82,
        evidence="一人称代词「我」→叙述者锚点",
        attribution_type=AttributionType.IMPLICIT,
    )


# ── P2：对话块检测 ────────────────────────────────────────────────────────────

# 两句台词之间旁白字数超过此值，视为新的对话块
_DIALOGUE_GAP_THRESHOLD = 150


def _build_dialogue_blocks(quotes: list[QuoteSpan]) -> dict[str, str]:
    """
    将 quotes 按文档位置分组为「对话块」。
    相邻两句之间间隙（旁白字数）< _DIALOGUE_GAP_THRESHOLD 时归为同一块。

    Returns:
        {quote_id: "对话块#N 第M/K句"}
        单句独立块（K=1）不生成标注（空字符串），避免噪音。
    """
    if not quotes:
        return {}

    # 逐句分组
    blocks: list[list[QuoteSpan]] = []
    current: list[QuoteSpan] = [quotes[0]]

    for i in range(1, len(quotes)):
        prev_end   = quotes[i - 1].raw_end   if quotes[i - 1].raw_end   is not None else quotes[i - 1].end
        curr_start = quotes[i].raw_start if quotes[i].raw_start is not None else quotes[i].start
        gap = curr_start - prev_end
        if gap <= _DIALOGUE_GAP_THRESHOLD:
            current.append(quotes[i])
        else:
            blocks.append(current)
            current = [quotes[i]]
    blocks.append(current)

    result: dict[str, str] = {}
    for block_id, block in enumerate(blocks, start=1):
        size = len(block)
        for pos, quote in enumerate(block, start=1):
            result[quote.quote_id] = "" if size == 1 else f"对话块#{block_id} 第{pos}/{size}句"
    return result


# ── P0：批量 LLM 归因路径 ────────────────────────────────────────────────────

def _parse_with_batch_llm(
    cleaned: str,
    quotes: list[QuoteSpan],
    aliases: AliasRegistry,
    ner_backend: NERBackend | None,
    batch_llm_config: Any,
    narrator: str | None,
    role_hints_list: list[str],
    return_result: bool,
    review_threshold: float,
    include_narration: bool,
    initial_recent_speakers: list[str] | None = None,
    on_progress: Any | None = None,
    enable_block_review: bool = True,
) -> list[SegmentEx] | ParseResult:
    """
    使用 BatchLLMAttributor 的完整归因路径。

    阶段：
      1. 生成所有候选集
      2. 预过滤（P1a 自我介绍 → P1b 叙述者锚点 → 保守规则层）
      3. 批量 LLM 归因（仅发送未解决台词）
      4. 合并归因，构建 SegmentEx 列表
      5. fix_consistency
    """
    from .batch_llm_attributor import BatchLLMAttributor, attribute_explicit_conservative

    # ── 章节边界检测 ─────────────────────────────────────────────────────────
    chapter_offsets = _detect_chapter_offsets(cleaned)  # 各章起始偏移（升序）
    if chapter_offsets:
        logger.info(f"[parser] 检测到 {len(chapter_offsets)} 个章节边界，将按章节隔离归因上下文")
    else:
        logger.info("[parser] 未检测到章节标题，全文共用归因上下文")

    def _quote_chapter_idx(quote: QuoteSpan) -> int:
        """返回该台词属于第几章（0 = 第一章或无章节头）。"""
        pos = quote.start
        idx = 0
        for off in chapter_offsets:
            if pos >= off:
                idx += 1
            else:
                break
        return idx

    # 预计算视角转换标志 + 各章叙述者
    # perspective_shift_chapters[i] = True  → 第 i 章是他人视角章节
    # chapter_narrators[i]                  → 第 i 章的叙述者（正常章节=主叙述者，视角转换章节=章节角色）
    n_chapters = len(chapter_offsets) + 1
    perspective_shift_chapters: list[bool] = [False] * n_chapters
    chapter_narrators: list[str | None] = [narrator] * n_chapters  # 默认全用主叙述者
    uncertain_chapters: list[dict] = []  # 叙述者身份不确定的章节，收录到 stats 供人工处理

    if narrator and chapter_offsets:
        _chapter_attributor = BatchLLMAttributor(batch_llm_config)
        first_next = chapter_offsets[0] if chapter_offsets else None
        if first_next is None or first_next == 0:
            perspective_shift_chapters[0] = False
        else:
            perspective_shift_chapters[0] = _is_perspective_shift_chapter(
                "", cleaned[:min(400, first_next)], narrator, role_hints=role_hints_list
            )
        for ci, off in enumerate(chapter_offsets):
            next_off = chapter_offsets[ci + 1] if ci + 1 < len(chapter_offsets) else None
            # off 可能指向行首 \n，跳过它找到标题文本起始位置
            h_start = off + 1 if off < len(cleaned) and cleaned[off] == "\n" else off
            h_end = cleaned.find("\n", h_start)
            heading = cleaned[h_start:h_end].strip() if h_end != -1 else cleaned[h_start:h_start + 80].strip()
            opening = _chapter_opening(cleaned, off, next_offset=next_off)
            # 剥离引号对话，只保留叙述文字，避免台词内容干扰视角判断
            opening_stripped = re.sub(r"「[^」]*」", "", opening).strip()

            # ── 前置过滤：不需要/无法 LLM 判断的章节 ─────────────────────────────
            # 1. 空标题或元段落（后记/插图/前言等）：不存在小说内视角，直接跳过
            _is_meta = not heading or any(kw in heading for kw in _META_SECTION_KEYWORDS)
            # 2. 开头叙述文字过短（纯标题页/插图页等无内容章节）：LLM 无法判断
            _too_short = len(opening_stripped) < 30
            if _is_meta or _too_short:
                reason_skip = "元段落" if _is_meta else f"叙述内容过短（{len(opening_stripped)}字）"
                logger.info(f"[parser] ch{ci + 1} 「{heading}」{reason_skip}，跳过视角检测")
                continue

            # 规则层标题提示（作为 hint 供 LLM 参考，不再作为最终判断依据）
            # 支持单角色「X的故事」和双角色「X与Y的故事」两种模式
            title_hint = _extract_chapter_narrator_hint(heading, role_hints_list)

            # LLM 视角判断（主路径）
            llm_result = _chapter_attributor.detect_chapter_narrator(
                chapter_opening=opening_stripped,
                main_narrator=narrator,
                role_hints=role_hints_list,
                heading=heading,
                title_hint=title_hint,
            )

            # LLM 置信度足够时直接采用；否则回退到启发式规则
            if llm_result.confidence >= 0.5:
                is_shift = llm_result.is_shift
            else:
                is_shift = _is_perspective_shift_chapter(
                    heading, opening, narrator, role_hints=role_hints_list
                )

            perspective_shift_chapters[ci + 1] = is_shift

            if is_shift:
                if llm_result.narrator and llm_result.confidence >= 0.65:
                    # 高置信度：直接使用 LLM 识别的叙述者
                    chapter_narrators[ci + 1] = llm_result.narrator
                    logger.info(
                        f"[parser] ch{ci + 1} 视角转换 → 叙述者={llm_result.narrator}"
                        f"（conf={llm_result.confidence:.2f}，{llm_result.reason}）"
                    )
                else:
                    title_fallback = _title_fallback_narrator(heading, narrator, role_hints_list)
                    # 叙述者不确定：优先使用标题弱回退，仍标记待人工处理；
                    # 若标题也无法给出可用角色，再回退到主叙述者。
                    chapter_narrators[ci + 1] = title_fallback or narrator
                    uncertain_chapters.append({
                        "chapter_idx": ci + 1,
                        "heading": heading,
                        "llm_confidence": llm_result.confidence,
                        "llm_reason": llm_result.reason,
                        "narrator_hint": llm_result.narrator or title_fallback or title_hint,
                        "fallback_narrator": title_fallback,
                    })
                    logger.warning(
                        f"[parser] ch{ci + 1} 视角转换但叙述者不确定"
                        f"（conf={llm_result.confidence:.2f}，{llm_result.reason}）→ "
                        f"暂用{title_fallback or '主叙述者'}兜底"
                    )
            else:
                if llm_result.confidence < 0.5:
                    # LLM 对"未转换"也不确定，记录待审查
                    uncertain_chapters.append({
                        "chapter_idx": ci + 1,
                        "heading": heading,
                        "llm_confidence": llm_result.confidence,
                        "llm_reason": llm_result.reason,
                        "narrator_hint": title_hint,
                    })
                    logger.warning(
                        f"[parser] ch{ci + 1} 视角判断不确定"
                        f"（conf={llm_result.confidence:.2f}，{llm_result.reason}）"
                    )
                else:
                    logger.info(
                        f"[parser] ch{ci + 1} 视角未转换（conf={llm_result.confidence:.2f}，"
                        f"{llm_result.reason}）"
                    )

        shift_count = sum(perspective_shift_chapters)
        if shift_count:
            shift_detail = ", ".join(
                f"ch{i}={chapter_narrators[i]}" for i, s in enumerate(perspective_shift_chapters) if s
            )
            logger.info(f"[parser] 检测到 {shift_count} 个视角转换章节：{shift_detail}")
        if uncertain_chapters:
            logger.warning(
                f"[parser] {len(uncertain_chapters)} 个章节叙述者不确定，需人工确认：" +
                ", ".join(f"ch{c['chapter_idx']}「{c['heading']}」" for c in uncertain_chapters)
            )

    # ① 生成所有候选集（章节边界处重置 recent_speakers，避免跨章上下文污染）
    recent_speakers_acc: list[str] = list(initial_recent_speakers or [])
    all_candidates: dict[str, CandidateSet] = {}
    prev_chapter_idx = -1
    for quote in quotes:
        ch_idx = _quote_chapter_idx(quote)
        if ch_idx != prev_chapter_idx and prev_chapter_idx != -1:
            recent_speakers_acc = []
        prev_chapter_idx = ch_idx
        ch_narrator = chapter_narrators[ch_idx] if ch_idx < len(chapter_narrators) else narrator
        cset = generate_candidates(quote, aliases=aliases, nlp_backend=ner_backend,
                                   recent_speakers=recent_speakers_acc, narrator=ch_narrator)
        all_candidates[quote.quote_id] = cset

    # ② 预过滤：纯规则层（高精确率，低召回率）
    pre_resolved: dict[str, Attribution] = {}
    unresolved_quotes: list[QuoteSpan] = []
    narrator_hint_quotes: set[str] = set()  # P1b 触发但不预解析的台词 id

    for quote in quotes:
        attr: Attribution | None = None
        ch_idx = _quote_chapter_idx(quote)
        is_shift = perspective_shift_chapters[ch_idx]

        # P1a：自我介绍（「我是X」「我X」模式）
        attr = _attribute_self_identified(quote, role_hints_list)

        # 保守规则层（前/后显性，必须 role_hints 中存在的角色）
        # 优先于 P1b 运行：若前文已明确指出说话人（如「真唯说道：」），
        # 该角色即使台词含「我」也不应被错误归因给叙述者。
        if attr is None:
            attr = attribute_explicit_conservative(quote, role_hints_list)

        # P1b：叙述者锚点（改为提示模式，不再预解析）
        # 使用该章节的实际叙述者（正常章节=主叙述者，视角转换章节=章节角色）。
        # 视角转换章节不再禁用 P1b，而是换用正确的叙述者进行锚定提示。
        ch_narrator = chapter_narrators[ch_idx]
        if attr is None and ch_narrator:
            if _attribute_narrator_anchored(quote, ch_narrator) is not None:
                narrator_hint_quotes.add(quote.quote_id)

        # P1c：发送消息上下文 → 该台词是叙述者发出的消息正文（预解析）
        # 触发条件：
        #   1. 前文含「我…（回/发/传/送）了（讯息/消息/短信）」模式
        #   2. 台词文本较短（≤30字，符合消息内容特征）
        # 典型场景：叙述者发了一条回复，下一段引号即是那条消息的内容。
        if attr is None and ch_narrator:
            cb_tail = (quote.context_before or "")[-100:]
            if (_MSG_SEND_RE.search(cb_tail) and len(quote.text or "") <= 30):
                attr = Attribution(
                    quote_id=quote.quote_id,
                    speaker=ch_narrator,
                    confidence=0.78,
                    evidence="发送消息上下文：叙述者发出的消息内容",
                    attribution_type=AttributionType.IMPLICIT,
                )

        if attr is not None:
            pre_resolved[quote.quote_id] = attr
        else:
            unresolved_quotes.append(quote)

    # ③ 批量 LLM 归因（按章节分组，各章独立传递说话人上下文）
    llm_attributions: dict[str, Attribution] = {}
    if unresolved_quotes:
        # P2：基于所有 quotes（含已预解决的）计算对话块，保留位置关系
        block_hints = _build_dialogue_blocks(quotes)

        attributor = BatchLLMAttributor(batch_llm_config)

        # 将未解决台词按章节分组（保持文档顺序），各组独立重置 recent_speakers
        chapter_groups: dict[int, list[QuoteSpan]] = {}
        for q in unresolved_quotes:
            ci = _quote_chapter_idx(q)
            chapter_groups.setdefault(ci, []).append(q)

        total_unresolved = len(unresolved_quotes)
        completed_count = 0

        for ci in sorted(chapter_groups.keys()):
            group = chapter_groups[ci]
            offset = completed_count
            total = total_unresolved

            def _make_progress(off: int, tot: int, cb: Any) -> Any:
                if cb is None:
                    return None
                def _cb(done: int, _tot: int) -> None:
                    cb(off + done, tot)
                return _cb

            # 使用该章节的专属叙述者（视角转换章节用章节角色，普通章节用主叙述者）
            ci_narrator = chapter_narrators[ci] if ci < len(chapter_narrators) else narrator

            group_result = attributor.attribute(
                group,
                all_candidates,
                role_hints=role_hints_list,
                block_hints=block_hints,
                narrator=ci_narrator,
                narrator_hints=narrator_hint_quotes,
                on_progress=_make_progress(offset, total, on_progress),
            )
            llm_attributions.update(group_result)
            completed_count += len(group)

    # ③.5 LLM 输出别名规范化：将 LLM 可能输出的别名（如「紫阳花」「紫阳花同学」）
    # 映射回 AliasRegistry 中的规范名（如「濑名紫阳花」），保证输出名字一致性。
    # 跳过特殊名（旁白/未知/众人及外貌描述型名字：非 alias_map 的短名直接保留）。
    _SKIP_CANON = {"旁白", "未知", "众人", "大家", "二人", "三人", "所有人"}
    if aliases.has_hints():
        for attr in llm_attributions.values():
            if attr.speaker not in _SKIP_CANON:
                canonical = aliases.canonicalize(attr.speaker)
                if canonical != attr.speaker:
                    attr.speaker = canonical

    # ③.5b 关系称谓置信度约束
    # 若 LLM 返回原始关系称谓（妹妹/哥哥 等）作为说话人，
    # 先尝试 alias 规范化（已在上方处理）；若仍为原始称谓（无别名映射），
    # 则将置信度压低至 ≤0.65 并追加"关系称谓待关联"标记，令其进入复核队列。
    # 有别名映射的情况在上方已转换为规范名，不受此约束。
    _RELATION_TERMS_SET = {
        "妹妹", "哥哥", "姐姐", "弟弟",
        "爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶",
    }
    for attr in llm_attributions.values():
        if attr.speaker in _RELATION_TERMS_SET:
            # 再次尝试规范化（兜底）
            canonical = aliases.canonicalize(attr.speaker)
            if canonical != attr.speaker:
                attr.speaker = canonical
            else:
                # 未映射到具体角色 → 压低置信度，标记待关联
                if attr.confidence > 0.65:
                    attr.confidence = 0.65
                note = "关系称谓待关联"
                attr.evidence = f"{attr.evidence}；{note}" if attr.evidence else note

    # ③.5c 临时角色场景保护
    # 当候选集中有朋友/少女/访客等未命名人物，而 LLM 仍选择仅由
    # narrator_anchor 注入的叙述者时，通常是“主角吸附”风险。保留建议
    # speaker 但降为待复核，避免高置信误判直接进入生产结果。
    for qid, attr in llm_attributions.items():
        _guard_batch_llm_attribution(attr, all_candidates.get(qid))

    # ④ 合并并按文档顺序构建 SegmentEx
    segments: list[SegmentEx] = []
    cursor = 0

    def _append_narrator_gap(gap_start: int, gap_end: int, prefix: str) -> None:
        """
        将 cleaned[gap_start:gap_end] 作为旁白处理，自动在章节标题处切分。
        章节标题本身作为独立旁白片段，前后文本分别处理。
        """
        boundaries = [off for off in chapter_offsets if gap_start <= off < gap_end]
        if not boundaries:
            _append_narrator_segments(segments, cleaned[gap_start:gap_end], prefix)
            return
        sub_start = gap_start
        for off in boundaries:
            if off > sub_start:
                _append_narrator_segments(segments, cleaned[sub_start:off], prefix)
            # 章节标题行
            h_start = off + 1 if off < len(cleaned) and cleaned[off] == "\n" else off
            h_end_pos = cleaned.find("\n", h_start)
            h_end = h_end_pos if h_end_pos != -1 else min(h_start + 120, len(cleaned))
            heading_text = cleaned[h_start:h_end].strip()
            if heading_text:
                segments.append(SegmentEx(
                    quote_id=f"{prefix}_ch_title",
                    speaker="旁白",
                    text=heading_text,
                    confidence=1.0,
                    evidence="章节标题",
                    attribution_type=AttributionType.NARRATOR,
                    candidates=["旁白"],
                    candidate_sources={"旁白": ["chapter_heading"]},
                    scene_characters=[],
                    context_before="",
                    context_after="",
                ))
            sub_start = h_end
        if sub_start < gap_end:
            _append_narrator_segments(segments, cleaned[sub_start:gap_end], prefix)

    for quote in quotes:
        raw_start = quote.raw_start if quote.raw_start is not None else quote.start
        raw_end   = quote.raw_end   if quote.raw_end   is not None else quote.end

        if include_narration and raw_start > cursor:
            _append_narrator_gap(cursor, raw_start, quote.quote_id)

        cset = all_candidates.get(quote.quote_id)

        if quote.quote_id in pre_resolved:
            attribution = pre_resolved[quote.quote_id]
        elif quote.quote_id in llm_attributions:
            attribution = llm_attributions[quote.quote_id]
        else:
            attribution = Attribution(
                quote_id=quote.quote_id,
                speaker="未知",
                confidence=0.20,
                evidence="归因失败（规则+LLM均无结果）",
                attribution_type=AttributionType.UNKNOWN,
            )

        segment = SegmentEx(
            quote_id=quote.quote_id,
            speaker=attribution.speaker,
            text=quote.text,
            addressee=getattr(attribution, "addressee", None) or None,
            confidence=attribution.confidence,
            evidence=attribution.evidence,
            attribution_type=attribution.attribution_type,
            candidates=cset.candidates if cset else [],
            candidate_sources=cset.candidate_sources if cset else {},
            scene_characters=cset.scene_characters if cset else [],
            context_before=quote.context_before,
            context_after=quote.context_after,
        )
        segments.append(segment)
        cursor = max(cursor, raw_end)

    if include_narration and cursor < len(cleaned):
        _append_narrator_gap(cursor, len(cleaned), "tail")

    segments = fix_consistency(segments, narrator=narrator)
    segments, special_perspective_stats = _apply_special_perspective_chapter_overrides(segments, role_hints_list)
    segments, scene_state_stats = _apply_scene_state_constraints(
        segments,
        role_hints_list,
        narrator=narrator,
        review_threshold=review_threshold,
        aliases=aliases,
    )
    segments, address_term_stats = apply_address_term_backcheck(segments, review_threshold=review_threshold)

    # ── 对话块级结构化复核（最后一趟，失败安全）────────────────────────────────
    block_review_stats: dict[str, Any] = {"mode": "block_review", "enabled": False}
    if enable_block_review:
        try:
            segments, block_review_stats = apply_block_review(
                segments,
                quotes,
                cleaned,
                batch_llm_config,
                narrator=narrator,
                role_hints=role_hints_list,
                aliases=aliases,
            )
            block_review_stats["enabled"] = True
        except Exception as exc:  # noqa: BLE001 - never let review break the main path
            logger.warning(f"[parser] block_review skipped due to error: {exc}")
            block_review_stats = {"mode": "block_review", "enabled": False, "error": repr(exc)}

    result = ParseResult(
        segments=segments,
        review_items=collect_review_items(segments, threshold=review_threshold),
        stats={
            "quote_count": len(quotes),
            "review_count": sum(1 for s in segments if s.confidence < review_threshold),
            "pre_resolved": len(pre_resolved),
            "llm_resolved": len(llm_attributions),
            "batch_llm_enabled": True,
            "uncertain_narrator_chapters": uncertain_chapters,
            "special_perspective_chapter": special_perspective_stats,
            "scene_state": scene_state_stats,
            "address_term_backcheck": address_term_stats,
            "block_review": block_review_stats,
        },
    )
    return result if return_result else result.segments


# ── 主入口 ────────────────────────────────────────────────────────────────────

def parse_novel(
    text: str,
    role_hints: dict[str, list[str] | str] | list[str] | None = None,
    llm_config: Any | None = None,
    batch_llm_config: Any | None = None,
    narrator: str | None = None,
    return_result: bool = False,
    review_threshold: float = 0.7,
    use_hanlp: bool = False,
    hanlp_model: str | None = None,
    nlp_backend: NERBackend | None = None,
    enable_alias_inference: bool = True,
    implicit_strategy: str = "heuristic",
    implicit_ranker: CandidateRanker | None = None,
    include_narration: bool = False,
    initial_recent_speakers: list[str] | None = None,
    on_progress: Any | None = None,
    enable_block_review: bool = True,
) -> list[SegmentEx] | ParseResult:
    """Parse Chinese novel text into speaker-attributed TTS segments.

    Args:
        text:               原始小说文本。
        role_hints:         已知角色名（列表或 {名: [别名]} 字典）。
        llm_config:         （兼容旧接口）逐条 LLM 配置，不再推荐使用。
        batch_llm_config:   BatchConfig 实例，启用后走批量 LLM 归因路径（推荐）。
        narrator:           主视角叙述者角色名，设置后台词中的「我」自动锚点到该角色。
                            需配合 role_hints 包含该角色名使用。
        return_result:      为 True 时返回 ParseResult，否则返回 list[SegmentEx]。
        review_threshold:   低于该置信度的片段被收集到 review_items。
        include_narration:  是否在 segments 中插入旁白片段。
        （其余参数控制 NER 后端和隐式归因策略，仅在非 batch_llm 路径下生效。）
    """
    # 解包 {"narrator": "...", "roles": {...}} 外层包装（外部 AI 角色提取结果常用此格式）
    role_hints, narrator = _unwrap_role_hints(role_hints, narrator)

    cleaned = normalize_text(text)
    aliases = AliasRegistry.from_role_hints(role_hints)
    ner_backend = nlp_backend or (HanLPNERBackend(hanlp_model) if use_hanlp else None)
    quotes = extract_quotes(cleaned)
    role_hints_list = _normalize_role_hints(role_hints)

    # ── 别名推断（通用，两条路径共用）────────────────────────────────────────
    inferred_aliases: dict[str, str] = {}
    if enable_alias_inference and quotes:
        observed_names: list[str] = []
        for quote in quotes:
            candidate_set = generate_candidates(quote, aliases=aliases, nlp_backend=ner_backend)
            observed_names.extend(candidate_set.candidates)
        inferred_aliases = aliases.update_inferred_aliases(observed_names)

    # ── 批量 LLM 路径（P0）────────────────────────────────────────────────────
    if batch_llm_config is not None:
        return _parse_with_batch_llm(
            cleaned=cleaned,
            quotes=quotes,
            aliases=aliases,
            ner_backend=ner_backend,
            batch_llm_config=batch_llm_config,
            narrator=narrator,
            role_hints_list=role_hints_list,
            return_result=return_result,
            review_threshold=review_threshold,
            include_narration=include_narration,
            initial_recent_speakers=initial_recent_speakers,
            on_progress=on_progress,
            enable_block_review=enable_block_review,
        )

    # ── 经典逐条路径（兼容旧接口）──────────────────────────────────────────────
    segments: list[SegmentEx] = []
    recent_speakers: list[str] = list(initial_recent_speakers or [])
    ranker = implicit_ranker
    if ranker is None and implicit_strategy == "llm_spc":
        if llm_config is None:
            raise RuntimeError("llm_config is required when implicit_strategy='llm_spc'")
        ranker = OpenAICompatibleSPCRanker(llm_config)

    cursor = 0
    for quote in quotes:
        raw_start = quote.raw_start if quote.raw_start is not None else quote.start
        raw_end   = quote.raw_end   if quote.raw_end   is not None else quote.end
        if include_narration and raw_start > cursor:
            _append_narrator_segments(segments, cleaned[cursor:raw_start], quote.quote_id)

        candidate_set = generate_candidates(
            quote,
            aliases=aliases,
            recent_speakers=recent_speakers,
            nlp_backend=ner_backend,
            narrator=narrator,
        )

        # P1a：自我介绍优先
        attribution = _attribute_self_identified(quote, role_hints_list)

        # P1b：叙述者锚点
        if attribution is None and narrator:
            attribution = _attribute_narrator_anchored(quote, narrator)

        # 规则显性归因
        if attribution is None:
            attribution = attribute_explicit(quote, aliases=aliases)

        if attribution is None:
            attribution = attribute_implicit(
                quote, candidate_set,
                recent_speakers=recent_speakers,
                ranker=ranker,
            )
        elif attribution.speaker in {"他", "她"}:
            if ranker is not None:
                attribution = ranker.rank(quote, candidate_set, recent_speakers=recent_speakers)
                attribution.attribution_type = AttributionType.LATENT
            elif recent_speakers:
                attribution.speaker = recent_speakers[-1]
                attribution.confidence = min(attribution.confidence, 0.72)
                attribution.evidence = f"{attribution.evidence}；代词承接最近说话人"

        segment = SegmentEx(
            quote_id=quote.quote_id,
            speaker=attribution.speaker,
            text=quote.text,
            addressee=getattr(attribution, "addressee", None) or None,
            confidence=attribution.confidence,
            evidence=attribution.evidence,
            attribution_type=attribution.attribution_type,
            candidates=candidate_set.candidates,
            candidate_sources=candidate_set.candidate_sources,
            scene_characters=candidate_set.scene_characters,
            context_before=quote.context_before,
            context_after=quote.context_after,
        )
        segments.append(segment)
        if segment.speaker not in {"旁白", "未知"}:
            recent_speakers.append(segment.speaker)
        cursor = max(cursor, raw_end)

    if include_narration and cursor < len(cleaned):
        _append_narrator_segments(segments, cleaned[cursor:], "tail")

    segments = fix_consistency(segments, narrator=narrator)
    segments, special_perspective_stats = _apply_special_perspective_chapter_overrides(segments, role_hints_list)
    segments, scene_state_stats = _apply_scene_state_constraints(
        segments,
        role_hints_list,
        narrator=narrator,
        review_threshold=review_threshold,
        aliases=aliases,
    )
    segments, address_term_stats = apply_address_term_backcheck(segments, review_threshold=review_threshold)
    result = ParseResult(
        segments=segments,
        review_items=collect_review_items(segments, threshold=review_threshold),
        stats={
            "quote_count": len(quotes),
            "review_count": sum(1 for item in segments if item.confidence < review_threshold),
            "hanlp_enabled": ner_backend is not None,
            "inferred_aliases": inferred_aliases,
            "implicit_strategy": implicit_strategy,
            "special_perspective_chapter": special_perspective_stats,
            "scene_state": scene_state_stats,
            "address_term_backcheck": address_term_stats,
        },
    )
    return result if return_result else result.segments
