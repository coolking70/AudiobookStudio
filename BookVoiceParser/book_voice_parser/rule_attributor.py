from __future__ import annotations

import re

from .alias_registry import AliasRegistry
from .schema import Attribution, AttributionType, QuoteSpan


SPEECH_VERBS = "喃喃道|嘀咕道|低声道|轻声道|笑道|冷笑道|喝道|骂道|叹道|缓缓开口|开口|心想|想道|说道|说|问道|问|答道|答|喊道|喊|叫道|叫|道"
NAME = r"[\u4e00-\u9fa5A-Za-z0-9_]{1,12}?"
ACTOR_NAME = r"[\u4e00-\u9fa5A-Za-z0-9_]{1,12}"
GROUP_NAMES = {"众人", "大家", "二人", "三人", "所有人"}
INNER_VERBS = ("心想", "想道")
GROUP_RE = r"众人|大家|二人|三人|所有人"
ACTION_SUFFIXES = ("皱眉", "点头", "摇头", "抬头", "低头", "缓缓", "看", "望", "笑", "哭", "说", "问", "答", "喊", "叫")


BEFORE_PATTERNS = [
    re.compile(rf"(?P<speaker>{GROUP_RE})\s*(?:齐声|一齐|同时)?\s*(?:{SPEECH_VERBS})[：:，,。\s]*$"),
    re.compile(rf"(?P<speaker>{NAME})\s*(?:对|向|冲)(?P<addressee>{NAME})\s*(?:{SPEECH_VERBS})[：:，,。\s]*$"),
    re.compile(rf"(?P<speaker>{ACTOR_NAME})(?:[^。！？!?\n「」]{{0,18}}?)[，,]\s*(?:{SPEECH_VERBS})[：:，,。\s]*$"),
    re.compile(rf"(?P<speaker>{ACTOR_NAME})\s*(?:{SPEECH_VERBS})[：:，,。\s]*$"),
]

AFTER_PATTERNS = [
    re.compile(rf"^[，,。\s]*(?P<speaker>{GROUP_RE})\s*(?:齐声|一齐|同时)?\s*(?:{SPEECH_VERBS})"),
    re.compile(rf"^[，,。\s]*(?P<speaker>{NAME})\s*(?:对|向|冲)(?P<addressee>{NAME})\s*(?:{SPEECH_VERBS})"),
    re.compile(rf"^[，,。\s]*(?P<speaker>{ACTOR_NAME})(?:[^。！？!?\n「」]{{0,18}}?)[，,]\s*(?:{SPEECH_VERBS})"),
    re.compile(rf"^[，,。\s]*(?P<speaker>{ACTOR_NAME})\s*(?:{SPEECH_VERBS})"),
]


def _last_clause(text: str, limit: int = 48) -> str:
    chunk = text[-limit:]
    parts = re.split(r"[。！？!?\n]", chunk)
    return parts[-1].strip() if parts else chunk.strip()


def _first_clause(text: str, limit: int = 48) -> str:
    chunk = text[:limit]
    quote_pos = chunk.find("「")
    sentence_end = re.search(r"[。！？!?\n]", chunk)
    end_pos = sentence_end.start() if sentence_end else -1
    if quote_pos != -1 and (end_pos == -1 or quote_pos < end_pos):
        return ""
    parts = re.split(r"[。！？!?\n]", chunk)
    return parts[0].strip() if parts else chunk.strip()


def _best_known_name(raw: str, aliases: AliasRegistry) -> str:
    cleaned = (raw or "").strip(" ，,。:：")
    known_names = aliases.match_names()
    for known in known_names:
        if cleaned == known or cleaned.startswith(known):
            return aliases.canonicalize(known)
    for marker in ("对", "向", "冲"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    for suffix in ACTION_SUFFIXES:
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[: -len(suffix)]
    for known in known_names:
        if cleaned == known or cleaned.startswith(known) or cleaned.endswith(known):
            return aliases.canonicalize(known)
    return aliases.canonicalize(cleaned)


def _type_for(speaker: str, evidence: str) -> AttributionType:
    if speaker in GROUP_NAMES:
        return AttributionType.GROUP
    if any(verb in evidence for verb in INNER_VERBS):
        return AttributionType.LATENT
    return AttributionType.EXPLICIT_NEAR


def _allowed_speaker(speaker: str, aliases: AliasRegistry) -> bool:
    if not aliases.has_hints():
        return True
    return speaker in GROUP_NAMES or aliases.canonicalize(speaker) in set(aliases.known_names())


def attribute_explicit(quote: QuoteSpan, aliases: AliasRegistry | None = None) -> Attribution | None:
    aliases = aliases or AliasRegistry()
    before = _last_clause(quote.context_before)
    after = _first_clause(quote.context_after)

    for pattern in BEFORE_PATTERNS:
        match = pattern.search(before)
        if match:
            speaker = _best_known_name(match.group("speaker"), aliases)
            if not _allowed_speaker(speaker, aliases):
                continue
            attr_type = _type_for(speaker, before)
            if attr_type == AttributionType.EXPLICIT_NEAR:
                attr_type = AttributionType.EXPLICIT_BEFORE
            return Attribution(
                quote_id=quote.quote_id,
                speaker=speaker,
                addressee=aliases.canonicalize(match.groupdict().get("addressee") or ""),
                confidence=0.96,
                evidence=before,
                attribution_type=attr_type,
            )

    for pattern in AFTER_PATTERNS:
        match = pattern.search(after)
        if match:
            speaker = _best_known_name(match.group("speaker"), aliases)
            if not _allowed_speaker(speaker, aliases):
                continue
            attr_type = _type_for(speaker, after)
            if attr_type == AttributionType.EXPLICIT_NEAR:
                attr_type = AttributionType.EXPLICIT_AFTER
            return Attribution(
                quote_id=quote.quote_id,
                speaker=speaker,
                addressee=aliases.canonicalize(match.groupdict().get("addressee") or ""),
                confidence=0.96,
                evidence=after,
                attribution_type=attr_type,
            )

    return None
