from __future__ import annotations

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .schema import AttributionType, SegmentEx


logger = logging.getLogger(__name__)


SKIP_SPEAKERS = {"", "旁白", "未知", "UNKNOWN"}
GENERIC_SPEAKERS = {
    "少女",
    "女孩子",
    "女性",
    "旁边的孩子",
    "未命名人物",
    "其他角色",
    "三位女性",
    "姐姐",
    "妹妹",
    "哥哥",
    "弟弟",
    "妈妈",
    "母亲",
    "爸爸",
    "父亲",
}
ADDRESS_SOURCE = "address_term_backcheck"
LOCAL_ADDRESS_SOURCE = "address_term_local_context"
ALIAS_SOURCE = "surface_alias_canonicalized"
AUDIO_TRACK_SOURCE = "audio_track_narrator"
SPECIAL_PERSPECTIVE_SOURCE = "special_perspective_chapter"
MIN_LEARN_CONFIDENCE = 0.85
MIN_AUTO_CONFIDENCE = 0.82
REVIEW_CONFIDENCE = 0.69
ADDRESS_SUFFIXES = (
    "亲",
    "桑",
    "酱",
    "小姐",
    "同学",
    "妈妈",
    "母亲",
    "姐姐",
    "妹妹",
    "哥哥",
    "弟弟",
    "さん",
    "ちゃん",
)
ADDRESS_TERM_RE = re.compile(
    r"([一-龥A-Za-z0-9_]{2,10}(?:亲|桑|酱|小姐|同学|妈妈|母亲|姐姐|妹妹|哥哥|弟弟)|"
    r"[A-Za-z0-9_一-龥]{1,10}(?:さん|ちゃん))"
)
GENERIC_TERMS = {"同学", "小姐", "妈妈", "母亲", "姐姐", "妹妹", "哥哥", "弟弟", "摄影师"}
GENERIC_RELATION_TERMS = {"妈妈", "母亲", "姐姐", "妹妹", "哥哥", "弟弟", "爸爸", "父亲"}
GENERIC_RELATION_RE = re.compile(r"(妈妈|母亲|姐姐|妹妹|哥哥|弟弟|爸爸|父亲)")
FIRST_PERSON_RE = re.compile(r"(?:^|[，,。！？!?\s「『])我(?:[，,。！？!?\s」』]|$|[一-龥])")
NARRATOR_AFTER_CUES = (
    "我这样说完",
    "我说完",
    "我如此",
    "我大声",
    "我用",
    "我认为",
    "即使我",
    "当有人提出",
    "像这种话",
)
SURFACE_ALIAS_SUFFIXES = ("同学", "小姐", "前辈", "老师")
SELF_REFERENCE_CUES = ("我是", "我叫", "叫我", "自称", "名字是", "我的名字", "名叫")
SELF_REFERENCE_CONTEXT_CUES = (
    "扮演",
    "原来的",
    "大家心中的",
    "所谓的",
    "名为",
    "这个人",
    "那个人",
)
MAX_DEBUG_EVENTS = 30


@dataclass
class AddressTermProfile:
    term: str
    speaker_counts: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.speaker_counts.values())

    def dominant(self) -> tuple[str, float, int]:
        if not self.speaker_counts:
            return "", 0.0, 0
        speaker, count = self.speaker_counts.most_common(1)[0]
        total = max(1, self.total)
        return speaker, count / total, count


def _append_evidence(seg: SegmentEx, note: str) -> None:
    if not note:
        return
    evidence = str(seg.evidence or "")
    if note in evidence:
        return
    seg.evidence = f"{evidence}；{note}" if evidence else note


def _add_candidate_source(seg: SegmentEx, speaker: str, source: str = ADDRESS_SOURCE) -> None:
    if not speaker or speaker in SKIP_SPEAKERS:
        return
    if speaker not in seg.candidates:
        seg.candidates = [speaker, *list(seg.candidates or [])]
    if speaker not in seg.scene_characters:
        seg.scene_characters = [speaker, *list(seg.scene_characters or [])]
    sources = dict(seg.candidate_sources or {})
    values = list(sources.get(speaker) or [])
    if source not in values:
        values.append(source)
    sources[speaker] = values
    seg.candidate_sources = sources


def _has_candidate_source(seg: SegmentEx, source: str) -> bool:
    for values in (seg.candidate_sources or {}).values():
        if source in (values or []):
            return True
    return False


def _is_special_perspective_locked(seg: SegmentEx) -> bool:
    evidence = str(seg.evidence or "")
    return (
        _has_candidate_source(seg, SPECIAL_PERSPECTIVE_SOURCE)
        or _has_candidate_source(seg, AUDIO_TRACK_SOURCE)
        or "特殊视角章节" in evidence
        or "音档视角" in evidence
    )


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _speaker_aliases(name: str) -> set[str]:
    name = str(name or "").strip()
    if not name:
        return set()
    aliases = {name}
    chinese = "".join(re.findall(r"[一-龥]+", name))
    for suffix in sorted(GENERIC_TERMS, key=len, reverse=True):
        if chinese.endswith(suffix) and len(chinese) > len(suffix):
            chinese = chinese[: -len(suffix)]
            aliases.add(chinese)
            break
    if chinese in GENERIC_TERMS:
        return {item for item in aliases if len(item) >= 2 and item not in GENERIC_TERMS}
    if len(chinese) >= 4:
        aliases.add(chinese[:2])
    if len(chinese) >= 3:
        aliases.add(chinese[-2:])
        aliases.add(chinese[-3:])
        aliases.add("小" + chinese[-3:])
        aliases.add("小" + chinese[-2:])
        aliases.add("小" + chinese[-3:-1])
        aliases.add("小" + chinese[-3])
    return {item for item in aliases if len(item) >= 2 and item not in GENERIC_TERMS}


def _is_canonical_name(name: str) -> bool:
    name = str(name or "").strip()
    if name in SKIP_SPEAKERS or name in GENERIC_SPEAKERS or name in GENERIC_TERMS:
        return False
    if len(name) < 3:
        return False
    if any(name.endswith(term) for term in GENERIC_RELATION_TERMS):
        return False
    if any(name.endswith(suffix) for suffix in SURFACE_ALIAS_SUFFIXES):
        return False
    return bool(re.search(r"[一-龥]", name))


def _surface_aliases_for_canonical(name: str) -> set[str]:
    name = str(name or "").strip()
    aliases = _speaker_aliases(name)
    chinese = "".join(re.findall(r"[一-龥]+", name))
    if len(chinese) >= 2:
        bases = {chinese[:2], chinese[-2:]}
        if len(chinese) >= 3:
            bases.add(chinese[-3:])
        for base in bases:
            if not base:
                continue
            aliases.add(base)
            aliases.add(f"{base}同学")
            aliases.add(f"{base}小姐")
            aliases.add(f"{base}前辈")
    if len(chinese) >= 3:
        aliases.add("小" + chinese[-2:])
        aliases.add("小" + chinese[-3:])
        aliases.add("小" + chinese[-3:-1])
        aliases.add("小" + chinese[-3])
    aliases.discard(name)
    return {item for item in aliases if len(item) >= 2 and item not in GENERIC_TERMS}


def _build_surface_alias_map(segments: list[SegmentEx]) -> dict[str, str]:
    names: set[str] = set()
    speaker_counts: Counter[str] = Counter()
    for seg in segments:
        names.update(str(item or "").strip() for item in [seg.speaker, *(seg.candidates or []), *(seg.scene_characters or [])])
        speaker = str(seg.speaker or "").strip()
        if speaker:
            speaker_counts[speaker] += 1
    preliminary = sorted((name for name in names if _is_canonical_name(name)), key=len, reverse=True)
    canonicals: list[str] = []
    for name in preliminary:
        is_surface_of_longer = any(
            other != name
            and (
                other.endswith(name)
                or (name.startswith("小") and len(name) > 1 and other.endswith(name[1:]))
            )
            for other in preliminary
        )
        if not is_surface_of_longer:
            canonicals.append(name)
    alias_to_canonicals: dict[str, set[str]] = defaultdict(set)
    for canonical in canonicals:
        for alias in _surface_aliases_for_canonical(canonical):
            if alias != canonical:
                alias_to_canonicals[alias].add(canonical)
    result: dict[str, str] = {}
    for alias, targets in alias_to_canonicals.items():
        if alias not in names:
            continue
        if len(targets) == 1:
            result[alias] = next(iter(targets))
            continue
        ranked = sorted(targets, key=lambda name: (speaker_counts[name], len(name)), reverse=True)
        if not ranked:
            continue
        best = ranked[0]
        second_count = speaker_counts[ranked[1]] if len(ranked) > 1 else 0
        if speaker_counts[best] >= max(3, second_count * 3):
            result[alias] = best
    return result


def canonicalize_surface_speakers(segments: list[SegmentEx]) -> tuple[list[SegmentEx], dict[str, object]]:
    alias_map = _build_surface_alias_map(segments)
    if not alias_map:
        return segments, {"normalized": 0, "aliases": {}}

    normalized = 0
    updated: list[SegmentEx] = []
    for seg in segments:
        copy = seg.model_copy()
        old_speaker = str(copy.speaker or "").strip()
        new_speaker = alias_map.get(old_speaker, old_speaker)
        if new_speaker != old_speaker:
            copy.speaker = new_speaker
            _append_evidence(copy, f"表面别名归并：{old_speaker}→{new_speaker}")
            normalized += 1

        copy.candidates = _dedupe([alias_map.get(str(item or "").strip(), str(item or "").strip()) for item in (copy.candidates or [])])
        copy.scene_characters = _dedupe(
            [alias_map.get(str(item or "").strip(), str(item or "").strip()) for item in (copy.scene_characters or [])]
        )
        sources: dict[str, list[str]] = {}
        for name, values in (copy.candidate_sources or {}).items():
            raw = str(name or "").strip()
            canonical = alias_map.get(raw, raw)
            merged = list(sources.get(canonical) or [])
            for value in values or []:
                if value not in merged:
                    merged.append(value)
            if canonical != raw and ALIAS_SOURCE not in merged:
                merged.append(ALIAS_SOURCE)
            sources[canonical] = merged
        copy.candidate_sources = sources
        updated.append(copy)

    return updated, {
        "normalized": normalized,
        "aliases": dict(sorted(alias_map.items())),
    }


def _known_name_aliases(segments: list[SegmentEx]) -> set[str]:
    aliases: set[str] = set()
    for seg in segments:
        for name in [seg.speaker, *(seg.candidates or []), *(seg.scene_characters or [])]:
            if not name or name in SKIP_SPEAKERS or name in GENERIC_SPEAKERS:
                continue
            aliases.update(_speaker_aliases(str(name)))
    return {item for item in aliases if len(item) >= 2}


def _name_alias_map(segments: list[SegmentEx]) -> dict[str, str]:
    result: dict[str, str] = {}
    names: set[str] = set()
    for seg in segments:
        for name in [seg.speaker, *(seg.candidates or []), *(seg.scene_characters or [])]:
            name = str(name or "").strip()
            if _is_canonical_name(name):
                names.add(name)
    for name in names:
        result[name] = name
        for alias in _speaker_aliases(name):
            result.setdefault(alias, name)
    return result


def _guess_primary_narrator(segments: list[SegmentEx]) -> str:
    counts: Counter[str] = Counter()
    for seg in segments:
        speaker = str(seg.speaker or "").strip()
        if speaker in SKIP_SPEAKERS or speaker in GENERIC_SPEAKERS:
            continue
        text = str(seg.text or "")
        evidence = str(seg.evidence or "")
        weight = 0.0
        if FIRST_PERSON_RE.search(text):
            weight += 1.0
        if "叙述者" in evidence or "一人称" in evidence:
            weight += 2.0
        try:
            weight *= max(0.5, float(seg.confidence or 0.0))
        except Exception:
            pass
        if weight > 0:
            counts[speaker] += weight
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _next_narration_text(segments: list[SegmentEx], index: int, *, max_ahead: int = 1) -> str:
    for offset in range(1, max_ahead + 1):
        pos = index + offset
        if pos >= len(segments):
            break
        seg = segments[pos]
        if str(seg.speaker or "").strip() in {"旁白", "", "UNKNOWN", "未知"}:
            return str(seg.text or "")
        if offset == 1:
            return ""
    return ""


def _nearest_local_non_narrator(segments: list[SegmentEx], index: int, narrator: str, *, window: int = 8) -> str:
    narrator_prefix = narrator[:2] if len(narrator) >= 2 else narrator

    def candidate_at(pos: int) -> list[str]:
        if pos < 0 or pos >= len(segments):
            return []
        found: list[str] = []
        speaker = str(segments[pos].speaker or "").strip()
        if _is_canonical_name(speaker) and speaker != narrator:
            found.append(speaker)
        for name in [*(segments[pos].candidates or []), *(segments[pos].scene_characters or [])]:
            name = str(name or "").strip()
            if _is_canonical_name(name) and name != narrator and name not in found:
                found.append(name)
        return found

    ranked: list[tuple[int, int, str]] = []
    for distance in range(1, window + 1):
        for pos in (index - distance, index + distance):
            for candidate in candidate_at(pos):
                local_text = " ".join(
                    str(getattr(segments[p], "text", "") or "")
                    + " "
                    + str(getattr(segments[p], "evidence", "") or "")
                    for p in range(max(0, pos - 1), min(len(segments), pos + 2))
                )
                score = 0
                if narrator_prefix and candidate.startswith(narrator_prefix):
                    score += 8
                if any(term in local_text for term in ("妹妹", "弟弟", "遥奈")):
                    score += 5
                ranked.append((-score, distance, candidate))
    if not ranked:
        return ""
    ranked.sort()
    return ranked[0][2]


def _resolve_alias(alias_map: dict[str, str], raw_name: str) -> str:
    raw_name = str(raw_name or "").strip()
    if not raw_name:
        return ""
    if raw_name in alias_map:
        return alias_map[raw_name]
    for alias, canonical in sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True):
        if raw_name.endswith(alias) or alias.endswith(raw_name):
            return canonical
    return ""


def _apply_narration_context_backcheck(
    segments: list[SegmentEx],
    stats: dict[str, object],
) -> set[int]:
    narrator = _guess_primary_narrator(segments)
    alias_map = _name_alias_map(segments)
    corrected: set[int] = set()
    if narrator:
        alias_map.setdefault(narrator, narrator)

    for index, seg in enumerate(segments):
        if not _is_dialogue_segment(seg):
            continue
        current = str(seg.speaker or "").strip()
        if "人工复核" in str(seg.evidence or "") or "manual" in str(seg.evidence or "").lower():
            continue
        next_narration = _next_narration_text(segments, index, max_ahead=1)
        if not next_narration:
            continue

        target = ""
        reason = ""
        voice_match = re.search(r"([一-龥]{2,5})的声音", next_narration[:40])
        if voice_match:
            target = _resolve_alias(alias_map, voice_match.group(1))
            reason = f"后续旁白标明「{voice_match.group(1)}的声音」"
        elif narrator and current != narrator:
            stripped = next_narration.lstrip(" ，,。！？!?")
            if any(stripped.startswith(cue) for cue in NARRATOR_AFTER_CUES):
                target = narrator
                reason = f"后续旁白以一人称承接：{stripped[:14]}"

        if not target or target == current or target in SKIP_SPEAKERS:
            continue
        old = current or "未知"
        seg.speaker = target
        seg.confidence = max(float(seg.confidence or 0.0), 0.91)
        seg.attribution_type = AttributionType.IMPLICIT
        _add_candidate_source(seg, target, LOCAL_ADDRESS_SOURCE)
        _append_evidence(seg, f"旁白上下文反推：{reason}，由 {old} 修正为 {target}")
        corrected.add(index)
        stats["context_corrected"] = int(stats.get("context_corrected") or 0) + 1
        _debug_event(
            stats,
            {
                "type": "context_corrected",
                "quote_id": str(seg.quote_id or ""),
                "from": old,
                "to": target,
                "reason": reason,
            },
        )
    return corrected


# 「姐姐/哥哥」作为称呼（呼语）而非普通名词的判定。
# 仅当确为"在喊对方"时，叙述者台词才可能其实是（向叙述者称呼者的）下位亲属在说话。
_VOCATIVE_OPENERS = "，。！？、：；…「『 \t～~啊喂嗳呐哎嗯欸诶呀哦喔噢"
_VOCATIVE_CLOSERS = "，。！？、：；…」』 \t～~你您啊呀哦喔噢"
# 这些前缀说明是"扮演/作为/称呼/的…姐姐"等名词用法，绝不是在喊对方
_VOCATIVE_BLOCK_PREFIX = (
    "扮演", "作为", "成为", "当成", "装作", "称作", "叫我", "叫他", "叫她",
    "当我", "当你", "的", "个", "像", "比", "这", "那", "当",
)


def _is_elder_vocative(text: str, term: str) -> bool:
    """text 中的 term（姐姐/哥哥）是否以"呼语/称呼对方"的形态出现。

    True 例：「姐姐，早安」「啊，姐姐。」「我说姐姐啊」
    False 例：「玩扮演姐姐游戏」「当紫阳花同学的姐姐」「叫我姐姐」「作为姐姐」
    """
    text = str(text or "")
    for m in re.finditer(re.escape(term), text):
        i, j = m.start(), m.end()
        prev2 = text[max(0, i - 2):i]
        prev3 = text[max(0, i - 3):i]
        if any(prev2.endswith(p) or prev3.endswith(p) for p in _VOCATIVE_BLOCK_PREFIX):
            continue
        after = text[j:j + 1]
        if after and after not in _VOCATIVE_CLOSERS:
            continue  # 后接非呼语字符（如「姐姐游戏/姐姐般」）→ 复合词，非呼语
        before = text[i - 1] if i > 0 else ""
        if i == 0 or before in _VOCATIVE_OPENERS or prev2 in ("我说", "跟你", "和你"):
            return True
    return False


def _apply_relation_vocative_backcheck(
    segments: list[SegmentEx],
    stats: dict[str, object],
) -> set[int]:
    narrator = _guess_primary_narrator(segments)
    if not narrator:
        return set()
    corrected: set[int] = set()
    elder_terms = {"姐姐", "哥哥"}
    for index, seg in enumerate(segments):
        if not _is_dialogue_segment(seg):
            continue
        current = str(seg.speaker or "").strip()
        if current != narrator:
            continue
        if "人工复核" in str(seg.evidence or "") or "manual" in str(seg.evidence or "").lower():
            continue
        terms = [term for term in GENERIC_RELATION_RE.findall(str(seg.text or "")) if term in elder_terms]
        if not terms:
            continue
        # 闸①：必须是"在喊对方"的呼语用法，排除「扮演/作为/的…姐姐」等名词用法
        vocative_terms = [t for t in terms if _is_elder_vocative(seg.text, t)]
        if not vocative_terms:
            stats["relation_vocative_blocked_nonvocative"] = int(stats.get("relation_vocative_blocked_nonvocative") or 0) + 1
            continue
        terms = vocative_terms
        target = _nearest_local_non_narrator(segments, index, narrator)
        if not target:
            stats["relation_vocative_blocked"] = int(stats.get("relation_vocative_blocked") or 0) + 1
            _append_evidence(seg, f"关系称谓负权重：主叙述者台词含「{terms[0]}」，但未找到可靠局部说话人")
            continue
        # 闸②：改判目标必须在当前场景（±2 段窗口的 scene_characters），否则不改
        scene_window: set[str] = set()
        for p in range(max(0, index - 2), min(len(segments), index + 3)):
            scene_window.update(str(x or "").strip() for x in (segments[p].scene_characters or []))
        if scene_window and target not in scene_window:
            stats["relation_vocative_blocked_offscene"] = int(stats.get("relation_vocative_blocked_offscene") or 0) + 1
            _append_evidence(seg, f"关系称谓负权重：「{terms[0]}」呼语，但候选改判目标 {target} 不在当前场景，保留原判")
            continue
        old = current
        seg.speaker = target
        seg.confidence = max(float(seg.confidence or 0.0), 0.92)
        seg.attribution_type = AttributionType.IMPLICIT
        _add_candidate_source(seg, target, LOCAL_ADDRESS_SOURCE)
        _append_evidence(seg, f"关系称谓局部反推：主叙述者台词称呼「{terms[0]}」，由 {old} 修正为 {target}")
        corrected.add(index)
        stats["relation_vocative_corrected"] = int(stats.get("relation_vocative_corrected") or 0) + 1
        _debug_event(
            stats,
            {
                "type": "relation_vocative_corrected",
                "quote_id": str(seg.quote_id or ""),
                "term": terms[0],
                "from": old,
                "to": target,
            },
        )
    return corrected


def _extract_address_terms(text: str, known_aliases: set[str], *, include_generic: bool = False) -> list[str]:
    text = str(text or "")
    terms: list[str] = []

    def add_term(term: str) -> None:
        term = str(term or "").strip()
        if not term:
            return
        if term in GENERIC_TERMS and (not include_generic or term not in GENERIC_RELATION_TERMS):
            return
        if term not in terms:
            terms.append(term)

    for match in ADDRESS_TERM_RE.finditer(text):
        term = match.group(1).strip()
        for suffix in ADDRESS_SUFFIXES:
            if not term.endswith(suffix):
                continue
            prefix = term[: -len(suffix)]
            for alias in sorted(known_aliases, key=len, reverse=True):
                if prefix.endswith(alias):
                    add_term(f"{alias}{suffix}")
                    break
            break
        add_term(term)
    if include_generic:
        for match in GENERIC_RELATION_RE.finditer(text):
            add_term(match.group(1))
    for alias in sorted(known_aliases, key=len, reverse=True):
        if len(alias) < 2 or alias in GENERIC_TERMS:
            continue
        if alias in text and alias not in terms:
            terms.append(alias)
    return terms


def _term_targets_speaker(term: str, speaker: str) -> bool:
    if not term or not speaker:
        return False
    plain = term
    for suffix in ADDRESS_SUFFIXES:
        if plain.endswith(suffix):
            plain = plain[: -len(suffix)]
            break
    if not plain:
        return False
    aliases = _speaker_aliases(speaker)
    if plain in aliases:
        return True
    if plain.startswith("小") and plain[1:] in aliases:
        return True
    return False


def _looks_like_self_reference(term: str, text: str) -> bool:
    term = str(term or "").strip()
    text = str(text or "")
    if not term or not text:
        return False
    for cue in SELF_REFERENCE_CUES:
        idx = text.find(cue)
        if idx < 0:
            continue
        window = text[idx: idx + 24]
        if term in window:
            return True
    for cue in SELF_REFERENCE_CONTEXT_CUES:
        idx = text.find(cue)
        if idx < 0:
            continue
        window_start = max(0, idx - 8)
        window = text[window_start: idx + 36]
        if term in window:
            return True
    if re.search(rf"{re.escape(term)}(?:亲|桑|酱|小姐|同学)?(?:就)?是神", text):
        return True
    return False


def _has_self_address_conflict(seg: SegmentEx, terms: list[str]) -> str:
    current = str(seg.speaker or "").strip()
    if not current or current in SKIP_SPEAKERS:
        return ""
    for term in terms:
        stripped_base_only = False
        for suffix in sorted(GENERIC_TERMS, key=len, reverse=True):
            if current.endswith(suffix) and term == current[: -len(suffix)]:
                # "摄影师姐姐" may naturally say "对摄影师很有礼貌";
                # the stripped occupation/relation base is not a direct address.
                stripped_base_only = True
                break
        if stripped_base_only:
            continue
        if _term_targets_speaker(term, current) and not _looks_like_self_reference(term, seg.text):
            return term
    return ""


def _debug_event(stats: dict[str, object], event: dict[str, object]) -> None:
    events = stats.setdefault("debug_events", [])
    if isinstance(events, list) and len(events) < MAX_DEBUG_EVENTS:
        events.append(event)


def _is_dialogue_segment(seg: SegmentEx) -> bool:
    speaker = str(seg.speaker or "").strip()
    quote_id = str(seg.quote_id or "")
    if speaker in SKIP_SPEAKERS or speaker == "旁白":
        return False
    if "_n" in quote_id or "_ch_title" in quote_id:
        return False
    if _is_special_perspective_locked(seg):
        return False
    return True


def _dialogue_block_bounds(segments: list[SegmentEx], index: int) -> tuple[int, int]:
    left = index
    while left - 1 >= 0 and _is_dialogue_segment(segments[left - 1]):
        left -= 1
    right = index
    while right + 1 < len(segments) and _is_dialogue_segment(segments[right + 1]):
        right += 1
    return left, right


def _nearest_alternative_speaker_in_block(
    segments: list[SegmentEx],
    index: int,
    term: str,
    *,
    prefer_direction: int = -1,
) -> str:
    current = str(segments[index].speaker or "").strip()
    left, right = _dialogue_block_bounds(segments, index)

    def scan(direction: int) -> str:
        pos = index + direction
        while left <= pos <= right:
            speaker = str(segments[pos].speaker or "").strip()
            if (
                speaker
                and speaker != current
                and speaker not in SKIP_SPEAKERS
                and speaker not in GENERIC_SPEAKERS
                and not _term_targets_speaker(term, speaker)
            ):
                return speaker
            pos += direction
        return ""

    first = scan(prefer_direction)
    if first:
        return first
    return scan(1 if prefer_direction < 0 else -1)


def _apply_iterative_dialogue_backcheck(
    segments: list[SegmentEx],
    seed_indices: set[int],
    known_aliases: set[str],
    stats: dict[str, object],
) -> set[int]:
    """Propagate high-confidence address fixes inside continuous dialogue blocks.

    This deliberately stops at narrator/meta segments. It only auto-corrects
    lines that independently contain a self-address conflict, so a bad seed
    cannot rewrite a whole scene by style alone.
    """

    if not seed_indices:
        return set()

    corrected: set[int] = set()
    queue = list(seed_indices)
    visited: set[tuple[int, int]] = set()
    while queue:
        seed = queue.pop(0)
        if seed < 0 or seed >= len(segments) or not _is_dialogue_segment(segments[seed]):
            continue
        left, right = _dialogue_block_bounds(segments, seed)
        stats["iterative_blocks_scanned"] = int(stats.get("iterative_blocks_scanned") or 0) + 1
        for direction in (-1, 1):
            pos = seed + direction
            while left <= pos <= right:
                key = (seed, pos)
                if key in visited:
                    pos += direction
                    continue
                visited.add(key)
                seg = segments[pos]
                terms = _extract_address_terms(seg.text, known_aliases, include_generic=True)
                conflict_term = _has_self_address_conflict(seg, terms)
                if not conflict_term:
                    stats["iterative_stopped_clean"] = int(stats.get("iterative_stopped_clean") or 0) + 1
                    break
                candidate = _nearest_alternative_speaker_in_block(
                    segments,
                    pos,
                    conflict_term,
                    prefer_direction=-direction,
                )
                if not candidate:
                    stats["iterative_blocked"] = int(stats.get("iterative_blocked") or 0) + 1
                    _append_evidence(seg, f"对白块迭代反推：{conflict_term} 指向当前 speaker，但未找到可靠相邻说话人")
                    break
                old = str(seg.speaker or "").strip() or "未知"
                seg.speaker = candidate
                seg.confidence = max(float(seg.confidence or 0.0), 0.93)
                seg.attribution_type = AttributionType.IMPLICIT
                _add_candidate_source(seg, candidate, LOCAL_ADDRESS_SOURCE)
                _append_evidence(seg, f"对白块迭代反推：台词称呼「{conflict_term}」而非自称，由 {old} 修正为 {candidate}")
                corrected.add(pos)
                queue.append(pos)
                stats["iterative_corrected"] = int(stats.get("iterative_corrected") or 0) + 1
                _debug_event(
                    stats,
                    {
                        "type": "iterative_corrected",
                        "quote_id": str(seg.quote_id or ""),
                        "term": conflict_term,
                        "from": old,
                        "to": candidate,
                    },
                )
                pos += direction
    return corrected


def learn_address_term_profiles(
    segments: list[SegmentEx],
    min_confidence: float = MIN_LEARN_CONFIDENCE,
) -> dict[str, AddressTermProfile]:
    known_aliases = _known_name_aliases(segments)
    profiles: dict[str, AddressTermProfile] = {}
    for seg in segments:
        if _is_special_perspective_locked(seg):
            continue
        speaker = str(seg.speaker or "").strip()
        if speaker in SKIP_SPEAKERS or speaker in GENERIC_SPEAKERS:
            continue
        try:
            confidence = float(seg.confidence or 0.0)
        except Exception:
            confidence = 0.0
        evidence = str(seg.evidence or "")
        if confidence < min_confidence and "人工" not in evidence and "manual" not in evidence.lower():
            continue
        for term in _extract_address_terms(seg.text, known_aliases, include_generic=True):
            if _term_targets_speaker(term, speaker):
                continue
            profile = profiles.setdefault(term, AddressTermProfile(term=term))
            profile.speaker_counts[speaker] += 1
            if len(profile.examples) < 3:
                profile.examples.append(str(seg.quote_id or ""))
    return profiles


def _term_score(profile: AddressTermProfile) -> tuple[str, float]:
    speaker, ratio, count = profile.dominant()
    if not speaker:
        return "", 0.0
    has_address_suffix = any(profile.term.endswith(suffix) for suffix in ADDRESS_SUFFIXES)
    if count < 3:
        return speaker, 0.0
    if not has_address_suffix and (count < 5 or ratio < 0.75):
        return speaker, 0.0
    if count >= 3 and ratio >= 0.60:
        return speaker, min(0.95, 0.78 + 0.05 * count + 0.10 * ratio)
    if count >= 2 and ratio >= 0.67:
        return speaker, min(0.90, 0.74 + 0.06 * count + 0.08 * ratio)
    return speaker, 0.0


def _is_implicit_like(seg: SegmentEx) -> bool:
    attr_type = str(seg.attribution_type or "")
    evidence = str(seg.evidence or "")
    return (
        attr_type in {AttributionType.IMPLICIT.value, AttributionType.LATENT.value, AttributionType.UNKNOWN.value}
        or "LLM" in evidence
        or "隐式" in evidence
        or "关系称谓" in evidence
        or "场景约束" in evidence
        or "跳变" in evidence
    )


def _can_auto_correct(seg: SegmentEx, speaker: str, score: float, term: str) -> bool:
    if not speaker or speaker in SKIP_SPEAKERS:
        return False
    if score < MIN_AUTO_CONFIDENCE:
        return False
    current = str(seg.speaker or "").strip()
    has_address_suffix = any(str(term or "").endswith(suffix) for suffix in ADDRESS_SUFFIXES)
    if current in GENERIC_SPEAKERS or current in SKIP_SPEAKERS:
        if not has_address_suffix and len(str(term or "")) < 3:
            return False
        return True
    if str(term or "") in GENERIC_RELATION_TERMS and score >= 0.86 and _is_implicit_like(seg):
        return True
    nickname_address = str(term or "").startswith("小") and _term_targets_speaker(term, current)
    if score >= 0.90 and _term_targets_speaker(term, current):
        return True
    if score >= 0.90 and (has_address_suffix or nickname_address) and _term_targets_speaker(term, current):
        return True
    return False


def _mark_neighbors_for_review(segments: list[SegmentEx], index: int) -> int:
    marked = 0
    for neighbor_index in (index - 1, index + 1):
        if neighbor_index < 0 or neighbor_index >= len(segments):
            continue
        neighbor = segments[neighbor_index]
        if str(neighbor.speaker or "") in SKIP_SPEAKERS:
            continue
        if not _is_implicit_like(neighbor):
            continue
        try:
            confidence = float(neighbor.confidence or 0.0)
        except Exception:
            confidence = 0.0
        if confidence <= REVIEW_CONFIDENCE:
            continue
        neighbor.confidence = REVIEW_CONFIDENCE
        _append_evidence(neighbor, "相邻片段受称呼反推修正影响，建议复核")
        marked += 1
    return marked


def _local_self_address_candidate(segments: list[SegmentEx], index: int, term: str) -> str:
    current = str(segments[index].speaker or "").strip()
    if not term:
        return ""
    # Prefer the following same-address run: it usually continues the line that
    # was incorrectly absorbed by the addressee.
    left, right = _dialogue_block_bounds(segments, index)
    for direction in (1, -1):
        pos = index + direction
        while left <= pos <= right:
            seg = segments[pos]
            speaker = str(seg.speaker or "").strip()
            if not speaker or speaker == current or speaker in SKIP_SPEAKERS:
                pos += direction
                continue
            if speaker in GENERIC_SPEAKERS:
                pos += direction
                continue
            if term in str(seg.text or ""):
                return speaker
            pos += direction
    return ""


def apply_address_term_backcheck(
    segments: list[SegmentEx],
    review_threshold: float = 0.7,
) -> tuple[list[SegmentEx], dict[str, object]]:
    updated, alias_stats = canonicalize_surface_speakers([seg.model_copy() for seg in segments])
    profiles = learn_address_term_profiles(updated)
    usable: dict[str, tuple[str, float, AddressTermProfile]] = {}
    for term, profile in profiles.items():
        speaker, score = _term_score(profile)
        if speaker and score > 0:
            usable[term] = (speaker, score, profile)

    stats: dict[str, object] = {
        "mode": "address_term_backcheck",
        "surface_aliases": alias_stats,
        "learned_terms": len(profiles),
        "usable_terms": len(usable),
        "scanned": 0,
        "candidates_added": 0,
        "corrected": 0,
        "blocked": 0,
        "neighbors_marked": 0,
        "iterative_blocks_scanned": 0,
        "iterative_corrected": 0,
        "iterative_blocked": 0,
        "iterative_stopped_clean": 0,
        "context_corrected": 0,
        "relation_vocative_corrected": 0,
        "relation_vocative_blocked": 0,
        "debug_events": [],
        "terms": {
            term: {
                "speaker": speaker,
                "score": round(score, 3),
                "count": profile.speaker_counts[speaker],
                "total": profile.total,
            }
            for term, (speaker, score, profile) in sorted(usable.items())
        },
    }

    known_aliases = _known_name_aliases(updated)
    corrected_indices: set[int] = set()
    for index, seg in enumerate(updated):
        if _is_special_perspective_locked(seg):
            continue
        current = str(seg.speaker or "").strip()
        if current in {"旁白"}:
            continue
        evidence = str(seg.evidence or "")
        if "人工复核" in evidence or "manual" in evidence.lower():
            continue
        all_terms = _extract_address_terms(seg.text, known_aliases, include_generic=True)
        matched_terms = [
            term
            for term in all_terms
            if term in usable
            and not (_term_targets_speaker(term, current) and _looks_like_self_reference(term, seg.text))
        ]
        self_address_term = _has_self_address_conflict(seg, all_terms)
        if not matched_terms and not self_address_term:
            continue
        stats["scanned"] = int(stats["scanned"]) + 1
        if matched_terms:
            best_term = max(matched_terms, key=lambda item: usable[item][1])
            speaker, score, profile = usable[best_term]
        else:
            best_term = self_address_term
            speaker, score, profile = "", 0.0, None
        local_speaker = _local_self_address_candidate(updated, index, self_address_term or best_term)
        if self_address_term and local_speaker:
            speaker = local_speaker
            score = max(score, 0.93)
        if speaker == current:
            _add_candidate_source(seg, speaker)
            continue
        if speaker and _term_targets_speaker(best_term, speaker):
            continue

        if speaker:
            _add_candidate_source(seg, speaker, LOCAL_ADDRESS_SOURCE if speaker == local_speaker else ADDRESS_SOURCE)
            stats["candidates_added"] = int(stats["candidates_added"]) + 1
            if speaker == local_speaker:
                note = f"称呼反推：{best_term} 在相邻同称呼块中指向 {speaker}({score:.2f})"
            else:
                note = f"称呼反推：{best_term} 常由 {speaker} 使用({score:.2f})"
        else:
            note = f"称呼负权重：台词称呼当前 speaker「{best_term}」，但未发现自称语境"
        if speaker and _can_auto_correct(seg, speaker, score, best_term):
            old = current or "未知"
            seg.speaker = speaker
            seg.confidence = max(float(seg.confidence or 0.0), min(0.92, score))
            seg.attribution_type = AttributionType.IMPLICIT
            _append_evidence(seg, f"{note}，由 {old} 修正为 {speaker}")
            stats["corrected"] = int(stats["corrected"]) + 1
            corrected_indices.add(index)
            _debug_event(
                stats,
                {
                    "type": "address_corrected",
                    "quote_id": str(seg.quote_id or ""),
                    "term": best_term,
                    "from": old,
                    "to": speaker,
                    "score": round(score, 3),
                },
            )
            stats["neighbors_marked"] = int(stats["neighbors_marked"]) + _mark_neighbors_for_review(updated, index)
        else:
            # Learned address profiles are helpful candidates, but common names
            # such as "玲奈子" may be used by several people. Only a direct
            # self-address conflict is strong enough to demote a high-confidence
            # attribution automatically.
            if self_address_term and float(seg.confidence or 0.0) >= review_threshold:
                seg.confidence = REVIEW_CONFIDENCE
            if speaker:
                suffix = "，与当前 speaker 冲突，待人工确认" if self_address_term else "，作为候选保留"
                _append_evidence(seg, f"{note}{suffix}")
            else:
                _append_evidence(seg, f"{note}，待人工确认")
            stats["blocked"] = int(stats["blocked"]) + 1

    iterative_indices = _apply_iterative_dialogue_backcheck(updated, corrected_indices, known_aliases, stats)
    corrected_indices.update(iterative_indices)
    context_indices = _apply_narration_context_backcheck(updated, stats)
    corrected_indices.update(context_indices)
    relation_indices = _apply_relation_vocative_backcheck(updated, stats)
    corrected_indices.update(relation_indices)
    stats["corrected_quote_ids"] = [
        str(updated[index].quote_id or "")
        for index in sorted(corrected_indices)
        if 0 <= index < len(updated)
    ][:MAX_DEBUG_EVENTS]
    logger.info(
        "[address-term-backcheck] scanned=%s corrected=%s iterative_corrected=%s blocked=%s candidates=%s",
        stats.get("scanned"),
        stats.get("corrected"),
        stats.get("iterative_corrected"),
        stats.get("blocked"),
        stats.get("candidates_added"),
    )

    return updated, stats
