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

# 关系/家人称谓：在角色正式姓名未出现于上下文时，「妹妹/哥哥」等代称也能作为候选人。
# 这使 LLM 可以正确识别场景中以关系代称出现（而非全名）的角色，
# 并在用户配置了 alias（如"妹妹"→"甘织遥奈"）时自动映射到规范名。
_RELATION_TERMS: frozenset[str] = frozenset([
    "妹妹", "哥哥", "姐姐", "弟弟",
    "爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶",
])

DIRTY_PREFIXES = ("看见", "看向", "望见", "听见")
DIRTY_FRAGMENTS = ("此事", "不能", "知道", "打算", "一个人", "都别", "门外", "可否", "借一步", "带着")
ACTION_SUFFIXES = ("皱眉", "点头", "摇头", "抬头", "低头", "冷静", "缓缓", "看", "望", "笑", "哭")
STOP_CANDIDATES = {"他", "她", "它", "问", "说", "道", "答", "喊", "叫", "走", "看", "望", "笑", "哭", "站", "坐"}
FALLBACK_CANDIDATES = {"旁白", "未知"}
GROUP_CANDIDATES = {"众人", "大家", "二人", "三人", "所有人"}
TEMPORARY_CANDIDATES = {
    "朋友", "朋友A", "朋友B", "朋友C",
    "少女", "女孩子", "旁边的孩子", "对面的少女",
    "访客", "金发女性", "另一个女孩",
}


# 向下关系：relation term = named character 本身（如「妹妹」= 甘织遥奈）
# 只有这类才能通过「附近出现了谁」来推断说话人。
# 向上/平级关系（妈妈/爸爸/姐姐/哥哥）不做推断：附近出现的是 owner（女儿），
# 而非说话人（妈妈）——需通过显式 RelationRole 配置来处理。
_DOWNWARD_RELATIONS: frozenset[str] = frozenset(["妹妹", "弟弟"])


def _infer_relation_character(rel: str, wide_context: str, aliases: AliasRegistry) -> str | None:
    """
    仅处理向下关系称谓（妹妹/弟弟 = named character 本身）的上下文推断。

    策略1：±15 字强关联（如「妹妹遥奈」「遥奈，妹妹说」）
    策略2：±50 字句内弱关联（如「妹妹走过来……遥奈对我说」）

    向上/平级关系（妈妈/爸爸/姐姐/哥哥）返回 None：
    上下文中找到的是 owner（子女/弟妹），而非说话人本身，推断方向相反会导致误归因。
    这类需通过 AliasRegistry 中的 RelationRole 配置处理（owner 条件激活）。
    """
    if rel not in _DOWNWARD_RELATIONS:
        return None
    if not aliases.has_hints():
        return None

    for radius in (15, 50):
        for m in re.finditer(re.escape(rel), wide_context):
            window = wide_context[max(0, m.start() - radius): m.end() + radius]
            for role in aliases.known_names():
                short_forms: list[str] = [role]
                if len(role) >= 3:
                    short_forms.append(role[-2:])
                if len(role) >= 4:
                    short_forms.append(role[-3:])
                for sf in short_forms:
                    if sf in window:
                        return role

    return None


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
    if (
        not aliases.has_hints()
        or name in FALLBACK_CANDIDATES
        or name in GROUP_CANDIDATES
        or name in TEMPORARY_CANDIDATES
    ):
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


def _add_temporary_scene_candidates(
    names: list[str],
    sources: dict[str, list[str]],
    context: str,
    aliases: AliasRegistry,
) -> None:
    """Add unnamed-but-present people as candidates.

    These labels are intentionally conservative. They prevent BatchLLM from
    forcing an unnamed friend/visitor/girl into the narrator or a famous role
    simply because the canonical role list has no exact speaker.
    """
    if (
        any(token in context for token in ("她们", "三人", "三个人", "一群", "剩下的两个人"))
        and any(token in context for token in ("朋友", "会长", "女孩子", "少女"))
    ):
        for label in ("朋友A", "朋友B", "朋友C"):
            _add_candidate(names, sources, label, "temporary_scene", aliases)

    descriptor_patterns = [
        ("旁边的孩子", ("旁边的孩子", "旁边的女孩子", "旁边的女孩")),
        ("对面的少女", ("对面的少女", "坐在她对面的少女", "对面的女孩子")),
        ("女孩子", ("女孩子", "女孩")),
        ("少女", ("少女",)),
        ("访客", ("访客", "客人", "来客")),
        ("金发女性", ("金发女性", "金发的女性", "金发美女")),
        ("另一个女孩", ("另一个女孩", "另一个女孩子")),
    ]
    for label, patterns in descriptor_patterns:
        if any(pattern in context for pattern in patterns):
            _add_candidate(names, sources, label, "temporary_scene", aliases)


def generate_candidates(
    quote: QuoteSpan,
    aliases: AliasRegistry | None = None,
    recent_speakers: list[str] | None = None,
    nlp_backend: NERBackend | None = None,
    max_candidates: int = 8,
    narrator: str | None = None,
) -> CandidateSet:
    aliases = aliases or AliasRegistry()
    recent_speakers = recent_speakers or []
    context = f"{quote.context_before}\n{quote.context_after}"

    names: list[str] = []
    sources: dict[str, list[str]] = {}
    # P3 补丁：role_hints 上下文匹配同时检查全名和后2/3字简称
    # （日式长名常在正文中以「真唯」「玲奈子」等简称出现，全名匹配会漏掉）
    # 注意：关系角色（如「遥奈妈妈」）跳过此处，由下方 relation terms 块按 owner 条件激活。
    # 原因：「遥奈妈妈」的末2字「妈妈」是关系称谓，会误匹配任何含「妈妈」的上下文。
    for role in aliases.known_names():
        if aliases.is_relation_role(role):
            continue  # 关系角色不参与短名匹配，由 owner 条件块处理
        short_forms: list[str] = [role]
        if len(role) >= 3:
            short_forms.append(role[-2:])   # 后2字
        if len(role) >= 4:
            short_forms.append(role[-3:])   # 后3字
        if any(sf in context for sf in short_forms):
            _add_candidate(names, sources, role, "role_hints", aliases)

    # 外貌/身份描述别名匹配：4字以上的长别名直接在上下文中搜索，
    # 用于识别在名字揭示前以外貌描述出现的角色（如「亮色头发的女孩」→ 甘织遥奈）。
    # 短别名（≤3字）已由上方规范名简称覆盖，此处仅处理长别名避免误匹配。
    already_matched = set(names)
    for alias, canonical in aliases.alias_map.items():
        if len(alias) >= 4 and alias != canonical and canonical not in already_matched:
            if alias in context:
                _add_candidate(names, sources, canonical, "appearance_alias", aliases)
                already_matched.add(canonical)
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

    _add_temporary_scene_candidates(names, sources, context, aliases)

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

    # 关系/家人称谓候选提取（短窗口检测 + owner 条件激活）
    #
    # 来源强度分级（用于 prompt 显示区分与后处理置信度约束）：
    #   relation_conditional —— 有 RelationRole 配置且 owner 在宽上下文中出现 → 强信号
    #   relation_inferred    —— 旧式平铺别名（妹妹→甘织遥奈）或向下关系上下文推断 → 中高可信
    #   relation_mention     —— 仅称谓出现，无法关联具体角色 → 弱参考，在 prompt 中独立注释
    wide_ctx = f"{(quote.context_before or '')[-150:]}\n{(quote.context_after or '')[:150]}"
    short_ctx = f"{(quote.context_before or '')[-100:]}\n{(quote.context_after or '')[:100]}"
    for rel in _RELATION_TERMS:
        if rel not in short_ctx:
            continue

        relation_roles = aliases.get_relation_roles(rel)

        if relation_roles:
            # 有显式 RelationRole 配置 → owner 条件激活
            for rr in relation_roles:
                owner_forms: list[str] = [rr.owner]
                if len(rr.owner) >= 3:
                    owner_forms.append(rr.owner[-2:])
                if len(rr.owner) >= 4:
                    owner_forms.append(rr.owner[-3:])
                owner_in_scene = any(f in wide_ctx for f in owner_forms)

                if owner_in_scene:
                    # owner 在场 → 主候选，高强度
                    if rr.canonical not in names and rr.canonical != narrator:
                        names.append(rr.canonical)
                    src_list = sources.setdefault(rr.canonical, [])
                    if "relation_conditional" not in src_list:
                        src_list.append("relation_conditional")
                elif len(relation_roles) == 1:
                    # 唯一可能的 owner 但不在场 → 弱候选（比完全不加要好）
                    if rr.canonical not in names and rr.canonical != narrator:
                        names.append(rr.canonical)
                    src_list = sources.setdefault(rr.canonical, [])
                    if "relation_mention" not in src_list:
                        src_list.append("relation_mention")
                # 多 owner 且当前 owner 不在场 → 不加，避免候选噪音

        else:
            # 无 RelationRole 配置 → 回退旧逻辑
            canonical = aliases.canonicalize(rel)
            alias_resolved = (canonical != rel)  # 旧式平铺别名（如"妹妹"→"甘织遥奈"）

            if alias_resolved:
                # 旧式别名已给出映射，追加 relation_inferred 标记
                if canonical != narrator and _is_plausible_candidate(canonical):
                    if canonical not in names:
                        names.append(canonical)
                    src_list = sources.setdefault(canonical, [])
                    if "relation_inferred" not in src_list:
                        src_list.append("relation_inferred")
            else:
                # 无任何配置：仅向下关系可尝试上下文推断（妹妹/弟弟 = named char 本身）
                inferred = _infer_relation_character(rel, wide_ctx, aliases)
                if inferred and inferred != narrator:
                    if inferred not in names:
                        names.append(inferred)
                    src_list = sources.setdefault(inferred, [])
                    if "relation_inferred" not in src_list:
                        src_list.append("relation_inferred")
                elif _is_plausible_candidate(canonical) and canonical not in names and canonical != narrator:
                    # 最终兜底：原始称谓，弱参考
                    names.append(canonical)
                    sources.setdefault(canonical, []).append("relation_mention")

    # narrator 始终保留在候选集中（叙述者视角下其名字不会出现在自己的上下文窗口里）
    if narrator and narrator not in names:
        names.append(narrator)
        sources.setdefault(narrator, []).append("narrator_anchor")

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
