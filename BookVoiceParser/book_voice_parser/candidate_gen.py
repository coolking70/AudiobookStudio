from __future__ import annotations

import re
from collections import Counter

from .alias_registry import AliasRegistry
from .nlp_backend import NERBackend
from .rule_attributor import NAME, SPEECH_VERBS
from .schema import CandidateSet, QuoteSpan


ACTION_CUES = "推门|推开|走进|走出|走向|看见|看向|望见|望向|走|看|望|笑|哭|站|坐|点头|摇头"
CUE_RE = re.compile(rf"(?P<name>{NAME})\s*(?:{SPEECH_VERBS}|{ACTION_CUES})")
TITLE_RE = re.compile(r"(师兄|师姐|师妹|师父|殿下|陛下|公子|姑娘|小姐|夫人|大人|先生)")
GROUP_CUE_RE = re.compile(r"(众人|大家|二人|三人|所有人)\s*(?:齐声|一齐|同时)?\s*(?:说|说道|道|喊|喊道|答|答道)")
DIRTY_PREFIXES = ("看见", "看向", "望见", "听见")
DIRTY_FRAGMENTS = ("此事", "不能", "知道", "打算", "一个人", "都别", "门外", "可否", "借一步", "带着")
ACTION_SUFFIXES = ("皱眉", "点头", "摇头", "抬头", "低头", "冷静", "缓缓", "看", "望", "笑", "哭")
STOP_CANDIDATES = {"他", "她", "它", "问", "说", "道", "答", "喊", "叫", "走", "看", "望", "笑", "哭", "站", "坐"}
FALLBACK_CANDIDATES = {"旁白", "未知"}
GROUP_CANDIDATES = {"众人", "大家", "二人", "三人", "所有人"}


def _dedupe(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        cleaned = (name or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _normalize_candidate_name(name: str, aliases: AliasRegistry) -> str:
    cleaned = (name or "").strip(" ，,。:：")
    for prefix in DIRTY_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    for marker in ("对", "向", "冲", "从"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    for suffix in ("齐声", "一齐", "同时"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    for suffix in ACTION_SUFFIXES:
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[: -len(suffix)]
    return aliases.canonicalize(cleaned)


def _is_plausible_candidate(name: str) -> bool:
    if not name or name in STOP_CANDIDATES or len(name) > 8:
        return False
    return not any(fragment in name for fragment in DIRTY_FRAGMENTS)


def _is_allowed_by_hints(name: str, aliases: AliasRegistry) -> bool:
    if not aliases.has_hints() or name in FALLBACK_CANDIDATES or name in GROUP_CANDIDATES:
        return True
    return aliases.canonicalize(name) in set(aliases.known_names())


def _add_candidate(names: list[str], sources: dict[str, list[str]], name: str, source: str, aliases: AliasRegistry) -> None:
    cleaned = (name or "").strip()
    cleaned = aliases.canonicalize(cleaned)
    if not _is_allowed_by_hints(cleaned, aliases):
        return
    if not _is_plausible_candidate(cleaned):
        return
    names.append(cleaned)
    source_list = sources.setdefault(cleaned, [])
    if source not in source_list:
        source_list.append(source)


def generate_candidates(
    quote: QuoteSpan,
    aliases: AliasRegistry | None = None,
    recent_speakers: list[str] | None = None,
    nlp_backend: NERBackend | None = None,
    max_candidates: int = 8,
) -> CandidateSet:
    aliases = aliases or AliasRegistry()
    recent_speakers = recent_speakers or []
    context = f"{quote.context_before}\n{quote.context_after}"

    names: list[str] = []
    sources: dict[str, list[str]] = {}
    # P3 补丁：role_hints 上下文匹配同时检查全名和后2/3字简称
    # （日式长名常在正文中以「真唯」「玲奈子」等简称出现，全名匹配会漏掉）
    for role in aliases.known_names():
        short_forms: list[str] = [role]
        if len(role) >= 3:
            short_forms.append(role[-2:])   # 后2字
        if len(role) >= 4:
            short_forms.append(role[-3:])   # 后3字
        if any(sf in context for sf in short_forms):
            _add_candidate(names, sources, role, "role_hints", aliases)
    for name in recent_speakers[-4:]:
        _add_candidate(names, sources, name, "recent_speakers", aliases)
    for match in GROUP_CUE_RE.finditer(context):
        _add_candidate(names, sources, match.group(1), "group_cue", aliases)
    # P3: 仅在没有 role_hints 时才用 CUE_RE/TITLE_RE 从上下文抽取名字。
    # 有 role_hints 时，CUE_RE 会把「对着真唯说道」中的受话人「真唯」误列为候选，
    # 且这些名字已由上面的 role_hints 上下文匹配覆盖，重复抽取反而降低排名可靠性。
    if not aliases.has_hints():
        for match in CUE_RE.finditer(context):
            _add_candidate(names, sources, _normalize_candidate_name(match.group("name"), aliases), "rule_cue", aliases)
        for match in TITLE_RE.finditer(context):
            _add_candidate(names, sources, match.group(1), "title", aliases)
    if nlp_backend is not None:
        for name in nlp_backend.extract_person_names(context):
            _add_candidate(names, sources, _normalize_candidate_name(name, aliases), "hanlp_ner", aliases)

    # P4: 隐式兜底 —— 有 role_hints 但上下文无角色名匹配时（典型隐式台词），
    # 注入全量 recent_speakers（扩大窗口）和全部 role_hints，给 LLM 完整候选空间。
    real_so_far = [n for n in _dedupe(names) if n not in FALLBACK_CANDIDATES and n not in GROUP_CANDIDATES]
    if aliases.has_hints() and not real_so_far:
        for name in recent_speakers:                      # 扩大到全量历史，不限 4 条
            _add_candidate(names, sources, name, "recent_speakers_extended", aliases)
        for role in aliases.known_names():
            _add_candidate(names, sources, role, "role_hints_fallback", aliases)
        # 允许更多候选位，让 LLM 有完整选项
        max_candidates = max(max_candidates, len(aliases.known_names()) + 2)

    counts = Counter(names)
    before = quote.context_before
    after = quote.context_after

    def proximity(name: str) -> int:
        before_pos = before.rfind(name)
        after_pos = after.find(name)
        distances: list[int] = []
        if before_pos != -1:
            distances.append(len(before) - before_pos)
        if after_pos != -1:
            distances.append(after_pos)
        return min(distances) if distances else 10_000

    ordered = sorted(_dedupe(names), key=lambda name: (proximity(name), -counts[name], names.index(name)))
    candidates = _dedupe([*ordered, "旁白", "未知"])[:max_candidates]
    for fallback in ("旁白", "未知"):
        sources.setdefault(fallback, ["fallback"])
    scene_characters = [name for name in candidates if name not in {"旁白", "未知"}]
    return CandidateSet(
        quote_id=quote.quote_id,
        candidates=candidates,
        candidate_sources={name: sources.get(name, []) for name in candidates},
        scene_characters=scene_characters,
    )
