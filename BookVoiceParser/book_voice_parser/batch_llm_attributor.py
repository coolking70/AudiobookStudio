"""
batch_llm_attributor.py
=======================
批量 LLM 说话人归因模块。

设计原则
--------
1. 每次请求携带 8-12 条台词 + 角色列表 + 前后文，LLM 一次性返回所有归因。
   相比逐条调用，API 调用量降低 ~10x，且 LLM 能在批内看到对话轮次关系。

2. 规则层仅负责"极高置信度的显性模式"（confidence ≥ 0.92 才输出），
   其余全部由此模块处理。

3. 失败降级：LLM 调用出错或 JSON 解析失败时，逐条回退到 spc_ranker 单条模式，
   最终无法归因则标记为 speaker="未知", confidence=0.2。

4. 支持流式进度回调（on_progress），便于前端展示进度。

用法::

    from book_voice_parser.batch_llm_attributor import BatchLLMAttributor, BatchConfig
    from book_voice_parser.schema import QuoteSpan, CandidateSet

    cfg = BatchConfig(base_url="http://127.0.0.1:1234/v1", model="qwen/qwen3.6-35b-a3b")
    attributor = BatchLLMAttributor(cfg)

    attributions = attributor.attribute(quotes, candidates_map, role_hints=["王冢真唯", "甘织玲奈子"])
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx

from .schema import Attribution, AttributionType, CandidateSet, QuoteSpan

logger = logging.getLogger(__name__)


# ── 配置 ──────────────────────────────────────────────────────────────────

MAX_COMPLETION_TOKENS = 131072
CHAPTER_NARRATOR_MAX_TOKENS = 4096
MIN_COMPLETION_TOKENS = 512

@dataclass
class BatchConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "lm-studio"
    model: str = "qwen/qwen3.6-35b-a3b"
    batch_size: int = 10             # 每批台词数量（日式轻小说推荐 8-12）
    max_tokens: int = 2048           # 批量输出预算；本地小上下文模型可降到 512-2048
    temperature: float = 0.0
    timeout: int = 180               # 秒，批处理比单条慢
    context_chars: int = 200         # 每条台词截取的前后文字符数
    output_mode: str = "compact"     # "verbose"（JSON 数组）或 "compact"（管道分隔行）
    max_retries: int = 2             # 解析失败后重试次数
    disable_thinking: bool = True    # 禁用 Qwen3/DeepSeek 等模型的扩展思考模式


@dataclass
class ChapterNarratorResult:
    """LLM 对章节叙述者身份的分析结果。"""
    narrator: str | None  # 叙述者全名；不确定时为 None
    confidence: float      # 0.0–1.0
    reason: str            # 简短推理说明（≤30字）
    is_shift: bool         # 相对于主叙述者是否发生视角转换


class EmptyLLMContentError(RuntimeError):
    """Raised when the provider returns only reasoning/metadata and no answer text."""


class BatchLLMBadRequestError(RuntimeError):
    """Raised for provider 400 errors that splitting/retrying will not fix reliably."""


# ── Prompt 构建 ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个中文/中译日式小说的说话人归因专家。判断每句台词最可能的说话人。

【优先级最高：前文末尾动作主语】
每条台词都有 [前文] 和 [紧前文(最重要)]（台词引号前最近约 40 字）。
若 [紧前文] 末尾出现「X 说/道/问/答/开口/笑道/喊道」等表达，X 即说话人，
置信度设 0.92+，不允许被候选排序或角色知名度覆盖。
例：「玲奈子深吸一口气，说道：」→ 玲奈子是说话人，即使王冢真唯是主角也不能改。

【关键区分：动作主语 vs 被提及对象】
以下两类上下文中出现的角色名，"不是"台词说话人：
  ① 对象/受话人：「对着真唯说道」「朝她问道」「冲玲奈子笑」→ 说话人另有其人
  ② 被观察者：「看见真唯站在那里」「眼见她点头」→ 说话人是观察者
真正的说话人是动作的主语，例如：
  「真唯压住长发，开口说道：」→ 真唯是说话人（主语）✓
  「我望着真唯，如此说道：」 → 我（叙述者）是说话人（主语），真唯只是被提及 ✓

【对话轮换】
- 每条台词都附有[上一条说话人]信息，若上一条已确认是A，当前无归属标记时应优先考虑B
- 连续多句相同说话人需要有支撑证据，不要仅因为"方便"就连续归给同一人
- 禁止"知名度偏差"：不要因为某角色是主角/更重要就优先选择，要依据证据

【对话块】
- 标有「对话块#N 第M/K句」的台词属于同一段连续对话（原文中紧密相邻）
- 同一对话块内，若无明确归属标记，应严格交替归因（A→B→A→B...）
- 每轮新的对话块重新判断首句归属，不要无根据地沿用上一块的顺序

【未命名人物】
若说话人是场景中以外貌/身份描述出现、尚无姓名的人物（如"亮发少女""鲍伯头女孩"），
且候选列表中无对应名字，可用不超过 6 字的外貌/身份描述作为 speaker。
仅在确定该人物不是叙述者/旁白且无法归给任何已知角色时使用。
若候选人中出现"朋友A/朋友B/朋友C/访客/对面的少女/旁边的孩子"等临时角色，
说明上下文确有未命名人物在场；应优先在这些临时角色中判断，不要强行归给叙述者或知名主角。

【输出格式】
必须输出合法 JSON 数组，每个对象包含：
  id, speaker, confidence(0.0-1.0), attribution_type, evidence(≤20字)
attribution_type: explicit_before | explicit_after | implicit | latent | group | unknown
不要输出任何解释，只输出 JSON 数组。不确定时 confidence 填低值，不要猜测。
"""

_SYSTEM_PROMPT_COMPACT = """\
你是一个中文/中译日式小说的说话人归因专家。判断每句台词最可能的说话人。

【优先级最高：前文末尾动作主语】
每条台词都有 [前文] 和 [紧前文(最重要)]（台词引号前最近约 40 字）。
若 [紧前文] 末尾出现「X 说/道/问/答/开口/笑道/喊道」等表达，X 即说话人，
置信度设 0.92+，不允许被候选排序或角色知名度覆盖。
例：「玲奈子深吸一口气，说道：」→ 玲奈子是说话人，即使王冢真唯是主角也不能改。

【关键区分：动作主语 vs 被提及对象】
以下两类上下文中出现的角色名，"不是"台词说话人：
  ① 对象/受话人：「对着真唯说道」「朝她问道」「冲玲奈子笑」→ 说话人另有其人
  ② 被观察者：「看见真唯站在那里」「眼见她点头」→ 说话人是观察者
真正的说话人是动作的主语，例如：
  「真唯压住长发，开口说道：」→ 真唯是说话人（主语）✓
  「我望着真唯，如此说道：」 → 我（叙述者）是说话人（主语），真唯只是被提及 ✓

【对话轮换】
- 每条台词都附有[上一条说话人]信息，若上一条已确认是A，当前无归属标记时应优先考虑B
- 连续多句相同说话人需要有支撑证据，不要仅因为"方便"就连续归给同一人
- 禁止"知名度偏差"：不要因为某角色是主角/更重要就优先选择，要依据证据

【对话块】
- 标有「对话块#N 第M/K句」的台词属于同一段连续对话（原文中紧密相邻）
- 同一对话块内，若无明确归属标记，应严格交替归因（A→B→A→B...）
- 每轮新的对话块重新判断首句归属，不要无根据地沿用上一块的顺序

【未命名人物】
若说话人是场景中以外貌/身份描述出现、尚无姓名的人物（如"亮发少女""鲍伯头女孩"），
且候选列表中无对应名字，可用不超过 6 字的外貌/身份描述作为 speaker。
仅在确定该人物不是叙述者/旁白且无法归给任何已知角色时使用。
若候选人中出现"朋友A/朋友B/朋友C/访客/对面的少女/旁边的孩子"等临时角色，
说明上下文确有未命名人物在场；应优先在这些临时角色中判断，不要强行归给叙述者或知名主角。

【输出格式】
每行一条，格式：序号|说话人|置信度|类型|依据(≤15字)
类型缩写：eb=explicit_before, ea=explicit_after, im=implicit, la=latent, gr=group, un=unknown
例：
1|王冢真唯|0.92|eb|前文「真唯说道」
2|甘织玲奈子|0.70|im|上句真唯，轮换
3|旁白|0.95|eb|无引号叙述
4|亮发少女|0.75|im|前文描述亮色发女孩靠近
5|朋友A|0.72|im|朋友群中一人发言
不要输出任何解释或多余文本，只输出上述格式的行。
"""

_USER_TEMPLATE = """\
角色列表：{role_list}
{narrator_line}
请对以下 {count} 条台词逐一归因，输出 JSON 数组：

{quotes_block}

输出示例：
[
  {{"id": "q001", "speaker": "王冢真唯", "confidence": 0.92, "attribution_type": "explicit_before", "evidence": "前文「真唯压住长发说道」"}},
  {{"id": "q002", "speaker": "甘织玲奈子", "confidence": 0.70, "attribution_type": "implicit", "evidence": "上句是真唯，此句轮换给玲奈子"}}
]
"""

_USER_TEMPLATE_COMPACT = """\
角色列表：{role_list}
{narrator_line}
请对以下 {count} 条台词逐一归因，每行输出：序号|说话人|置信度|类型|依据

{quotes_block}
"""

_QUOTE_ITEM_TEMPLATE = """\
--- 台词 {idx} | id: {quote_id} | 候选人: {candidates} | 上一句说话人: {prev_speaker}{block_hint} ---
[前文] {context_before}
[紧前文(最重要)] {context_before_tail}
[台词] {quote_text}
[后文] {context_after}
{addressee_hint}{narrator_hint}{relation_hint}"""


# ── 章节叙述者检测 Prompt ──────────────────────────────────────────────────
_CHAPTER_NARRATOR_SYSTEM = """\
你是中文小说第一人称视角分析专家。根据章节开头文字，判断该章节叙述者是否仍是已知主叙述者，或已切换到其他角色视角。

判断依据：
1. 若主叙述者姓名/简称在非引号叙述中被第三方提及（如「玲奈子走进了教室」）→ 视角切换，is_shift=true
2. 若叙述自然以「我」进行，且主叙述者未以第三方角色身份出现 → 仍是主叙述者视角，is_shift=false
3. 若切换视角：寻找谁以第一人称行动、内心独白、身体感知 → 确定新叙述者

输出严格合法 JSON（无注释无多余文本）：
{"narrator": "角色全名或null", "confidence": 0.00-1.00, "reason": "≤30字理由", "is_shift": true/false}

narrator 规则：
- is_shift=false：填主叙述者名，或 null（不确定）
- is_shift=true ：填新叙述者全名，不确定时填 null
- confidence<0.5 时 narrator 通常填 null
"""


def _detect_addressee_trap(context_before: str, role_hints: list[str]) -> str:
    """
    检测前文中是否存在「对着/朝/望着/看向 + 角色名 + 说话动词」的受话人陷阱。
    如果检测到，返回提示字符串；否则返回空字符串。

    这个检测帮助 LLM 避免把「对着真唯说道」中的"真唯"误认为说话人。
    """
    ADDRESSEE_PATTERNS = re.compile(
        r"(?:对着?|朝|冲|向|望着?|看向|注视着?)\s*"
        r"(?P<name>[一-龥]{2,6})"
        r"(?:[一-龥，,\s]{0,15}?)"
        r"(?:说道|说|道|问道|问|答道|喊道)"
    )
    cb = context_before[-120:]
    m = ADDRESSEE_PATTERNS.search(cb)
    if m:
        name = m.group("name")
        # 确认这个名字是已知角色（避免误报）
        for role in role_hints:
            if name == role or role.endswith(name) or role.startswith(name):
                return f"⚠️注意：前文「{m.group().strip()}」中「{name}」是受话对象，不是说话人"
    return ""


def _build_prompt(
    batch: list[tuple[QuoteSpan, CandidateSet]],
    role_hints: list[str],
    context_chars: int,
    prev_speakers: list[str] | None = None,
    block_hints: dict[str, str] | None = None,
    output_mode: str = "compact",
    narrator: str | None = None,
    narrator_hints: set[str] | None = None,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)

    Args:
        block_hints:    {quote_id: "对话块#N 第M/K句"} —— 由 parser 层传入，注入提示词。
        output_mode:    "compact"（管道分隔行）或 "verbose"（JSON 数组）
        narrator:       主视角叙述者姓名，用于生成叙述者角色提示行。
        narrator_hints: 含「我」且 P1b 认为可能是叙述者的 quote_id 集合；
                        对应台词会在 prompt 中附加💡提示，由 LLM 最终确认。
    """
    role_list = "、".join(role_hints) if role_hints else "（未指定，从上下文推断）"
    prev_speakers = prev_speakers or []
    block_hints = block_hints or {}
    narrator_hints = narrator_hints or set()

    # 叙述者角色提示行（注入 user template）
    if narrator:
        narrator_line = (
            f"（叙述者是「{narrator}」，叙述段落中「我」通常指她；"
            f"但对话场景里其他角色也可能说「我」，标有💡的台词请结合上下文判断）"
        )
    else:
        narrator_line = "（第一人称叙述者「我」通常映射到角色列表中的某一固定角色，请结合上下文判断）"

    quotes_block_parts = []
    for i, (quote, cset) in enumerate(batch):
        cb = (quote.context_before or "")[-context_chars:]
        cb_tail = cb[-40:]  # 紧前文：最靠近台词的 40 字，是最重要的归因线索
        ca = (quote.context_after or "")[:context_chars]

        # 将候选人按来源强度分组：
        #   relation_mention（仅称谓，无 owner 关联）→ 独立提示行（弱参考）
        #   其余（role_hints / relation_conditional / relation_inferred 等）→ 正常候选列表
        _STRONG_RELATION_SOURCES = {"relation_conditional", "relation_inferred"}
        _relation_mention_only: list[str] = []
        _main_candidates: list[str] = []
        for c in (cset.candidates or []):
            if c in {"旁白", "未知"}:
                continue
            c_sources = set(cset.candidate_sources.get(c, []))
            if "relation_mention" in c_sources and not (c_sources & _STRONG_RELATION_SOURCES):
                _relation_mention_only.append(c)
            else:
                _main_candidates.append(c)

        candidates_str = "、".join(_main_candidates)
        if not candidates_str:
            candidates_str = role_list

        # 上一条说话人（批内 i-1，或从外部传入的 prev_speakers[-1]）
        if i > 0:
            prev_sp = prev_speakers[i - 1] if i - 1 < len(prev_speakers) else "未知"
        elif prev_speakers:
            prev_sp = prev_speakers[-1]
        else:
            prev_sp = "（批次首条）"

        # 对话块标注（P2）
        bh = block_hints.get(quote.quote_id, "")
        block_hint_str = f" | {bh}" if bh else ""

        # 受话人陷阱检测
        addressee_hint = _detect_addressee_trap(quote.context_before or "", role_hints)

        # 叙述者候选提示（P1b demoted：含「我」但不确定说话人，提示 LLM 结合上下文判断）
        narrator_hint_str = ""
        if narrator and quote.quote_id in narrator_hints:
            narrator_hint_str = (
                f"\n💡叙述者参考：台词含「我」，叙述者是「{narrator}」，"
                f"但场景中其他人物也可能说「我」——请结合前后文确认实际说话人，勿强行归给叙述者。"
            )

        # 关系称谓弱参考提示（relation_mention：仅在上下文出现称谓，未关联具体角色）
        relation_hint_str = ""
        if _relation_mention_only:
            terms_str = "、".join(_relation_mention_only)
            relation_hint_str = (
                f"\n📌关系称谓参考：「{terms_str}」出现在上下文中，"
                f"请优先从候选人列表中找到对应的具体角色；"
                f"仅当确认说话人不在候选列表时，才以称谓本身标记。"
            )

        part = _QUOTE_ITEM_TEMPLATE.format(
            idx=i + 1,
            quote_id=quote.quote_id,
            candidates=candidates_str,
            prev_speaker=prev_sp,
            block_hint=block_hint_str,
            context_before=cb or "（无前文）",
            context_before_tail=cb_tail or "（无紧前文）",
            quote_text=quote.text,
            context_after=ca or "（无后文）",
            addressee_hint=addressee_hint,
            narrator_hint=narrator_hint_str,
            relation_hint=relation_hint_str,
        )
        quotes_block_parts.append(part)

    quotes_block = "\n".join(quotes_block_parts)
    if output_mode == "compact":
        user_prompt = _USER_TEMPLATE_COMPACT.format(
            role_list=role_list,
            narrator_line=narrator_line,
            count=len(batch),
            quotes_block=quotes_block,
        )
        return _SYSTEM_PROMPT_COMPACT, user_prompt
    else:
        user_prompt = _USER_TEMPLATE.format(
            role_list=role_list,
            narrator_line=narrator_line,
            count=len(batch),
            quotes_block=quotes_block,
        )
        return _SYSTEM_PROMPT, user_prompt


# ── JSON 解析 ──────────────────────────────────────────────────────────────

_ATTR_TYPE_MAP = {
    "explicit_before": AttributionType.EXPLICIT_BEFORE,
    "explicit_after":  AttributionType.EXPLICIT_AFTER,
    "implicit":        AttributionType.IMPLICIT,
    "latent":          AttributionType.LATENT,
    "group":           AttributionType.GROUP,
    "unknown":         AttributionType.UNKNOWN,
}


def _extract_json_array(text: str) -> list | None:
    """从 LLM 原始输出中提取 JSON 数组，支持思维链模型的多种输出格式。"""
    # 1. 剥离 <think>...</think> 块（完整或因 max_tokens 截断的残缺块）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)  # 残缺 <think>（无闭合标签）
    text = text.strip()

    # 2. 优先从代码围栏提取（```json ... ``` 或 ``` ... ```）
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 找文本中最后一个 [...] 块（思维链可能在前面产生伪数组）
    spans = [(m.start(), m.end()) for m in re.finditer(r"\[", text)]
    for start in reversed(spans):
        # 从每个 [ 开始往后找匹配的 ]
        depth = 0
        for i, ch in enumerate(text[start[0]:]):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start[0]: start[0] + i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # 这个 [ 不行，试上一个
    return None


_COMPACT_TYPE_MAP = {
    "eb": AttributionType.EXPLICIT_BEFORE,
    "ea": AttributionType.EXPLICIT_AFTER,
    "im": AttributionType.IMPLICIT,
    "la": AttributionType.LATENT,
    "gr": AttributionType.GROUP,
    "un": AttributionType.UNKNOWN,
    # 兼容全称
    "explicit_before": AttributionType.EXPLICIT_BEFORE,
    "explicit_after":  AttributionType.EXPLICIT_AFTER,
    "implicit":        AttributionType.IMPLICIT,
    "latent":          AttributionType.LATENT,
    "group":           AttributionType.GROUP,
    "unknown":         AttributionType.UNKNOWN,
}


def _parse_compact_response(text: str, batch: list[tuple[QuoteSpan, CandidateSet]]) -> list[Attribution]:
    """解析紧凑管道分隔格式的 LLM 输出：序号|说话人|置信度|类型|依据"""
    # 剥离思维链
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    text = text.strip()

    results: list[Attribution] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        idx_str, speaker = parts[0].strip(), parts[1].strip()
        # 容错：允许 "1." / "1、" / " 1 " 等变体，只取数字部分
        idx_digits = re.sub(r"\D", "", idx_str)
        if not idx_digits:
            continue
        idx = int(idx_digits) - 1
        if idx < 0 or idx >= len(batch):
            logger.debug(f"Compact: index {idx+1} out of range (batch size {len(batch)})")
            continue

        try:
            confidence = float(parts[2].strip())
        except ValueError:
            confidence = 0.5
        confidence = min(max(confidence, 0.0), 1.0)

        attr_str = parts[3].strip()
        attr_type = _COMPACT_TYPE_MAP.get(attr_str, AttributionType.IMPLICIT)

        evidence = parts[4].strip() if len(parts) > 4 else ""

        quote = batch[idx][0]
        results.append(Attribution(
            quote_id=quote.quote_id,
            speaker=speaker,
            confidence=confidence,
            evidence=evidence,
            attribution_type=attr_type,
        ))

    return results


def _parse_llm_response(text: str, batch: list[tuple[QuoteSpan, CandidateSet]]) -> list[Attribution]:
    """从 LLM 输出中解析归因结果，失败返回空列表。"""
    items = _extract_json_array(text)
    if items is None:
        logger.warning("LLM response contains no JSON array")
        return []
    if not isinstance(items, list):
        logger.warning(f"JSON root is not an array: {type(items)}")
        return []

    # 建 id→QuoteSpan 映射
    id_map = {quote.quote_id: quote for quote, _ in batch}

    results: list[Attribution] = []
    for item in items:
        qid = str(item.get("id", ""))
        # 如果 LLM 返回了序号而非 id，尝试按位置匹配
        if qid not in id_map and qid.isdigit():
            idx = int(qid) - 1
            if 0 <= idx < len(batch):
                qid = batch[idx][0].quote_id

        if qid not in id_map:
            # 尝试宽松匹配
            for bid, _ in id_map.items():
                if str(qid) in str(bid) or str(bid) in str(qid):
                    qid = bid
                    break
            else:
                logger.debug(f"Cannot match id={item.get('id')} to any quote")
                continue

        speaker    = str(item.get("speaker", "未知")).strip()
        confidence = float(item.get("confidence", 0.5))
        attr_str   = str(item.get("attribution_type", "implicit")).strip()
        evidence   = str(item.get("evidence", "")).strip()

        attr_type = _ATTR_TYPE_MAP.get(attr_str, AttributionType.IMPLICIT)

        results.append(Attribution(
            quote_id=qid,
            speaker=speaker,
            confidence=min(max(confidence, 0.0), 1.0),
            evidence=evidence,
            attribution_type=attr_type,
        ))

    return results


# ── 主类 ──────────────────────────────────────────────────────────────────

class BatchLLMAttributor:
    """批量 LLM 说话人归因器。"""

    def __init__(self, config: BatchConfig):
        self.cfg = config
        self._client = None

    def _normalize_base_url(self) -> str:
        base = str(self.cfg.base_url or "").strip().rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")].rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                self._client = False
            else:
                self._client = OpenAI(
                    base_url=self._normalize_base_url(),
                    api_key=self.cfg.api_key,
                )
        return self._client

    def _max_rate_limit_retries(self) -> int:
        return int(getattr(self.cfg, "rate_limit_retries", 6) or 0)

    @staticmethod
    def _is_rate_limited(status_code: int | None, exc: Exception | None = None) -> bool:
        if status_code == 429:
            return True
        if exc is not None:
            text = str(exc).lower()
            return "429" in text or "rate limit" in text
        return False

    @staticmethod
    def _rate_limit_delay(source: Any, attempt: int) -> float:
        # Honor Retry-After when present; else exponential backoff (5,10,20,40s) capped at 60s.
        headers = getattr(getattr(source, "response", None), "headers", None) or getattr(source, "headers", None)
        if headers:
            raw = headers.get("Retry-After") or headers.get("retry-after")
            try:
                if raw is not None:
                    return min(120.0, max(1.0, float(raw)))
            except (TypeError, ValueError):
                pass
        return min(60.0, 5.0 * (2 ** attempt))

    def _call_llm_via_httpx(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key or 'local'}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(
            connect=min(float(self.cfg.timeout), 30.0),
            read=float(self.cfg.timeout),
            write=60.0,
            pool=30.0,
        )
        max_retries = self._max_rate_limit_retries()
        attempt = 0
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            while True:
                response = client.post(
                    f"{self._normalize_base_url()}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 400:
                    raise BatchLLMBadRequestError(
                        f"HTTP 400 from LLM: {response.text[:800]}"
                    )
                if response.status_code == 429 and attempt < max_retries:
                    delay = self._rate_limit_delay(response, attempt)
                    logger.warning("BatchLLM HTTP 429 限流，%.0fs 后重试（第 %d/%d 次）", delay, attempt + 1, max_retries)
                    time.sleep(delay)
                    attempt += 1
                    continue
                response.raise_for_status()
                return response.json()

    def _resolve_max_tokens(self, max_tokens: int | None = None) -> int:
        """Return a completion-token budget accepted by common OpenAI-compatible servers."""
        raw = max_tokens if max_tokens is not None else self.cfg.max_tokens
        try:
            value = int(raw or 512)
        except (TypeError, ValueError):
            value = 512
        value = max(MIN_COMPLETION_TOKENS, value)
        if value > MAX_COMPLETION_TOKENS:
            logger.warning(
                "BatchLLM max_tokens=%s 超过接口常见上限，已钳制为 %s",
                value,
                MAX_COMPLETION_TOKENS,
            )
            value = MAX_COMPLETION_TOKENS
        return value

    def _call_llm(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """调用 LLM，返回原始文本。

        兼容思考模型（Qwen3 / MiMo / DeepSeek-R1 等）禁用思考的三种策略：
        1. extra_body enable_thinking=False（vLLM / Ollama 原生支持，LM Studio 可能忽略）
        2. assistant 前缀注入空思考块 <think>\\n\\n</think>（在 LM Studio 上最可靠）
           ——模型看到思考阶段"已完成"，直接输出答案，reasoning_tokens 降至 0
        3. /no_think user 前缀（部分 Qwen3 变体支持，LM Studio 通常忽略）
        策略 1 + 2 同时启用，互为兜底。

        Args:
            max_tokens: 覆盖 BatchConfig.max_tokens，用于轻量级调用（如章节叙述者检测）。
        """
        resolved_max_tokens = self._resolve_max_tokens(max_tokens)
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        extra_body: dict = {}
        if self.cfg.disable_thinking:
            normalized_base_url = self._normalize_base_url().lower()
            is_lm_studio = "127.0.0.1:1234" in normalized_base_url or "localhost:1234" in normalized_base_url
            supports_reasoning_options = (
                is_lm_studio
                or "siliconflow" in normalized_base_url
                or "api.siliconflow.cn" in normalized_base_url
                or "localhost" in normalized_base_url
                or "127.0.0.1" in normalized_base_url
            )
            # 策略 1：通过 extra_body 通知后端禁用思考（vLLM/Ollama 有效）
            if supports_reasoning_options:
                extra_body["enable_thinking"] = False
                # LM Studio 新版支持 reasoning_effort=none；对 Gemma/Qwen 本地模型比空 think 前缀更稳定。
                extra_body["reasoning_effort"] = "none"
            # 策略 2：注入空思考块作为 assistant 前缀（LM Studio 上最可靠）
            # 模型看到 <think>...</think> 已闭合，认为思考已完成，直接生成答案
            # 但新版 LM Studio 在支持 reasoning_effort=none 时不再需要该前缀；
            # 少一个 assistant prefill 也能节省上下文，降低 400/context exceeded 风险。
            if supports_reasoning_options and not is_lm_studio:
                messages.append({"role": "assistant", "content": "<think>\n\n</think>\n"})
            # 策略 3：Qwen3 系列常见的 no-think 指令。即使后端忽略，也不会改变任务语义。
            if supports_reasoning_options and not is_lm_studio:
                messages[1]["content"] = "/no_think\n" + str(messages[1]["content"])

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": resolved_max_tokens,
            "temperature": self.cfg.temperature,
        }

        client = self._get_client()
        if extra_body:
            if client:
                payload["extra_body"] = extra_body
            else:
                payload.update(extra_body)
        if client:
            max_retries = self._max_rate_limit_retries()
            attempt = 0
            while True:
                try:
                    resp = client.chat.completions.create(
                        **payload,
                        timeout=self.cfg.timeout,
                    )
                    break
                except Exception as exc:
                    status_code = getattr(exc, "status_code", None)
                    response = getattr(exc, "response", None)
                    status_code = status_code or getattr(response, "status_code", None)
                    if status_code == 400:
                        body = ""
                        try:
                            body = getattr(response, "text", "") or str(exc)
                        except Exception:
                            body = str(exc)
                        raise BatchLLMBadRequestError(f"HTTP 400 from LLM: {body[:800]}") from exc
                    if self._is_rate_limited(status_code, exc) and attempt < max_retries:
                        delay = self._rate_limit_delay(exc, attempt)
                        logger.warning("BatchLLM HTTP 429 限流，%.0fs 后重试（第 %d/%d 次）", delay, attempt + 1, max_retries)
                        time.sleep(delay)
                        attempt += 1
                        continue
                    raise
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
        else:
            data = self._call_llm_via_httpx(payload)
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            content = (content or "").strip()

        if not content:
            # 思考模型的 reasoning_content 回退（不标准，仅作诊断）
            try:
                extra = getattr(msg, "model_extra", None) or {}
                if isinstance(msg, dict):
                    extra = msg.get("model_extra") or msg.get("extra") or extra
                reasoning = extra.get("reasoning_content", "") if isinstance(extra, dict) else ""
                reasoning = reasoning or ""
                if reasoning:
                    logger.warning(
                        "content 为空，模型仅返回 reasoning_content；可能是 thinking 预算不足、"
                        "服务端内容过滤或模型模板未把最终答案写入 content。"
                        f"reasoning_content 长度={len(reasoning)} 字（≈{len(reasoning)//2} tokens），"
                        f"当前 max_tokens={resolved_max_tokens}，建议设为 "
                        f"{min(MAX_COMPLETION_TOKENS, max(resolved_max_tokens, len(reasoning)//2 + 2048))}"
                    )
            except Exception:
                pass
        return content

    @staticmethod
    def _unknown_attribution(quote: QuoteSpan, evidence: str = "LLM 空正文/疑似内容过滤") -> Attribution:
        return Attribution(
            quote_id=quote.quote_id,
            speaker="未知",
            confidence=0.2,
            evidence=evidence,
            attribution_type=AttributionType.UNKNOWN,
        )

    def _unknown_attributions_for_batch(self, batch: list[tuple[QuoteSpan, CandidateSet]], evidence: str) -> list[Attribution]:
        return [
            self._unknown_attribution(quote, evidence=evidence)
            for quote, _ in batch
        ]

    def detect_chapter_narrator(
        self,
        chapter_opening: str,
        main_narrator: str,
        role_hints: list[str],
        heading: str = "",
        title_hint: str | None = None,
    ) -> ChapterNarratorResult:
        """
        使用 LLM 判断章节叙述者。

        Args:
            chapter_opening: 章节开头文字（~400字，建议已剥离引号对话）
            main_narrator:   已知主叙述者全名
            role_hints:      全部已知角色名列表
            heading:         章节标题（可为空字符串）
            title_hint:      规则层从标题提取的叙述者名（仅作参考提示）

        Returns:
            ChapterNarratorResult
        """
        if not main_narrator:
            return ChapterNarratorResult(
                narrator=None, confidence=0.0, reason="无主叙述者信息", is_shift=False
            )

        roles_str = "、".join(role_hints) if role_hints else "（未知）"
        hint_line = ""
        if title_hint:
            hint_line = f"\n规则提示：标题「{heading}」含模式，疑似叙述者「{title_hint}」（仅供参考）。"

        user_prompt = (
            f"主叙述者：{main_narrator}\n"
            f"全部角色：{roles_str}\n"
            f"章节标题：{heading or '（无）'}{hint_line}\n\n"
            f"章节开头（对话已剥离）：\n{chapter_opening[:400]}\n\n"
            f"请分析叙述视角并输出 JSON。"
        )

        try:
            # 叙述者检测只需要小 JSON，但 thinking 模型若无法禁用思考，512 token
            # 容易全部耗在 reasoning_content，导致 content 为空。
            raw = self._call_llm(
                _CHAPTER_NARRATOR_SYSTEM,
                user_prompt,
                max_tokens=CHAPTER_NARRATOR_MAX_TOKENS,
            )
            # 剥离思维链残留
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            raw = re.sub(r"<think>.*$", "", raw, flags=re.DOTALL)
            raw = raw.strip()
            # 提取 JSON 对象（允许前后有多余文字）
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                narrator_val: str | None = obj.get("narrator")
                if isinstance(narrator_val, str):
                    narrator_val = narrator_val.strip() or None
                    # 兼容 LLM 返回字符串 "null" / "none"
                    if narrator_val and narrator_val.lower() in ("null", "none", "不确定"):
                        narrator_val = None
                else:
                    narrator_val = None
                confidence = float(obj.get("confidence", 0.5))
                confidence = min(max(confidence, 0.0), 1.0)
                reason = str(obj.get("reason", "")).strip()
                is_shift = bool(obj.get("is_shift", False))
                return ChapterNarratorResult(
                    narrator=narrator_val,
                    confidence=confidence,
                    reason=reason,
                    is_shift=is_shift,
                )
            logger.warning(f"detect_chapter_narrator: JSON not found in: {raw[:200]!r}")
        except Exception as e:
            logger.warning(f"detect_chapter_narrator failed: {e}")

        # 失败降级：假设视角未转换
        return ChapterNarratorResult(
            narrator=main_narrator,
            confidence=0.0,
            reason="LLM 失败，降级为主叙述者",
            is_shift=False,
        )

    def attribute_batch(
        self,
        batch: list[tuple[QuoteSpan, CandidateSet]],
        role_hints: list[str],
        prev_speakers: list[str] | None = None,
        block_hints: dict[str, str] | None = None,
        narrator: str | None = None,
        narrator_hints: set[str] | None = None,
    ) -> list[Attribution]:
        """
        对一批台词归因，返回 Attribution 列表。

        Args:
            batch:           (QuoteSpan, CandidateSet) 对的列表
            role_hints:      已知角色名列表
            prev_speakers:   批次前的最近若干说话人（用于对话轮换上下文）
            block_hints:     {quote_id: "对话块#N 第M/K句"}，注入 prompt（P2）
            narrator:        叙述者姓名，用于生成叙述者角色提示行
            narrator_hints:  P1b 触发的 quote_id 集合，注入💡提示
        """
        system, user = _build_prompt(
            batch, role_hints, self.cfg.context_chars, prev_speakers, block_hints,
            output_mode=self.cfg.output_mode,
            narrator=narrator,
            narrator_hints=narrator_hints,
        )
        parse_fn = _parse_compact_response if self.cfg.output_mode == "compact" else _parse_llm_response
        last_empty = False

        for attempt in range(self.cfg.max_retries + 1):
            try:
                raw = self._call_llm(system, user)
                if not raw.strip():
                    last_empty = True
                    raise EmptyLLMContentError("LLM 返回空 content")
                results = parse_fn(raw, batch)
                if results:
                    logger.debug(f"Batch {len(batch)} quotes → {len(results)} attributions")
                    return results
                if attempt < self.cfg.max_retries:
                    logger.warning(f"Empty parse result, retry {attempt+1}")
                    time.sleep(1)
            except Exception as e:
                if isinstance(e, BatchLLMBadRequestError):
                    logger.warning(
                        "BatchLLM 请求被 LLM 服务拒绝，已将本批 %s 条台词标记为未知待复核：%s",
                        len(batch),
                        e,
                    )
                    return self._unknown_attributions_for_batch(
                        batch,
                        evidence=f"LLM 400 Bad Request：{str(e)[:180]}",
                    )
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt < self.cfg.max_retries:
                    time.sleep(2)

        if len(batch) > 1:
            mid = max(1, len(batch) // 2)
            reason = "空 content" if last_empty else "空解析结果"
            logger.warning(
                "BatchLLM %s，已将 %s 条台词拆成 %s/%s 条以定位疑似过滤片段",
                reason,
                len(batch),
                mid,
                len(batch) - mid,
            )
            left = self.attribute_batch(
                batch[:mid],
                role_hints,
                prev_speakers=prev_speakers,
                block_hints=block_hints,
                narrator=narrator,
                narrator_hints=narrator_hints,
            )
            next_prev = (list(prev_speakers or []) + [a.speaker for a in left])[-4:]
            right = self.attribute_batch(
                batch[mid:],
                role_hints,
                prev_speakers=next_prev,
                block_hints=block_hints,
                narrator=narrator,
                narrator_hints=narrator_hints,
            )
            return left + right

        if batch:
            quote, _ = batch[0]
            logger.warning(
                "BatchLLM 单句仍无法解析，已标记为未知待复核：%s",
                quote.quote_id,
            )
            return [self._unknown_attribution(quote)]

        return []

    def attribute(
        self,
        quotes: list[QuoteSpan],
        candidates_map: dict[str, CandidateSet],
        role_hints: list[str] | None = None,
        block_hints: dict[str, str] | None = None,
        narrator: str | None = None,
        narrator_hints: set[str] | None = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, Attribution]:
        """
        对所有台词批量归因，批次间传递说话人上下文。

        Args:
            quotes:          QuoteSpan 列表（应按文档顺序排列）
            candidates_map:  quote_id → CandidateSet
            role_hints:      已知角色名列表
            block_hints:     {quote_id: "对话块#N 第M/K句"}，由 parser 层传入（P2）
            narrator:        叙述者姓名，传递给每批的 attribute_batch
            narrator_hints:  P1b 触发的 quote_id 集合，传递给每批的 attribute_batch
            on_progress:     进度回调 (已完成数, 总数)

        Returns:
            dict: quote_id → Attribution
        """
        role_hints = role_hints or []
        block_hints = block_hints or {}
        result: dict[str, Attribution] = {}
        total = len(quotes)

        # 分批（保持文档顺序，确保批次间上下文可传递）
        batches: list[list[tuple[QuoteSpan, CandidateSet]]] = []
        current_batch: list[tuple[QuoteSpan, CandidateSet]] = []
        for quote in quotes:
            cset = candidates_map.get(quote.quote_id, CandidateSet(
                quote_id=quote.quote_id,
                candidates=role_hints + ["旁白", "未知"],
                candidate_sources={},
                scene_characters=role_hints,
            ))
            current_batch.append((quote, cset))
            if len(current_batch) >= self.cfg.batch_size:
                batches.append(current_batch)
                current_batch = []
        if current_batch:
            batches.append(current_batch)

        completed = 0
        # 跨批次的最近说话人窗口（传给下一批的 prev_speakers）
        recent_speakers: list[str] = []

        for batch in batches:
            attributions = self.attribute_batch(
                batch, role_hints,
                prev_speakers=recent_speakers[-4:],
                block_hints=block_hints,
                narrator=narrator,
                narrator_hints=narrator_hints,
            )

            # 建 id→Attribution 映射
            attr_map = {a.quote_id: a for a in attributions}

            batch_speakers: list[str] = []
            for quote, _ in batch:
                if quote.quote_id in attr_map:
                    result[quote.quote_id] = attr_map[quote.quote_id]
                    batch_speakers.append(attr_map[quote.quote_id].speaker)
                else:
                    # LLM 未返回该 id 的结果 → 降级 unknown
                    logger.debug(f"No attribution for {quote.quote_id}, fallback to unknown")
                    result[quote.quote_id] = Attribution(
                        quote_id=quote.quote_id,
                        speaker="未知",
                        confidence=0.2,
                        evidence="LLM 未返回结果",
                        attribution_type=AttributionType.UNKNOWN,
                    )
                    batch_speakers.append("未知")

            # 更新跨批次上下文（保留最近 8 个）
            recent_speakers = (recent_speakers + batch_speakers)[-8:]

            completed += len(batch)
            if on_progress:
                on_progress(completed, total)
            logger.info(f"Progress: {completed}/{total}")

        return result


# ── 规则层保守模式（重构后的 attribute_explicit）────────────────────────────

import re as _re

# 说话动词：末尾有这些词才算显性
_SPEECH_VERB_END = _re.compile(
    r"(?:说道|说|道|问道|问|答道|答|喊道|喊|叫道|低声道|淡淡道|轻声道|喃喃道|"
    r"开口道|开口说|如此说|笑道|轻声说|缓缓道|补充道|说着)\s*[：:，,「\s]*$"
)
# 后显性：台词结束后角色名+动词紧跟（窗口更短，只看 40 字）
_AFTER_NAME_VERB = _re.compile(
    r"^[，,。！？」\s]{0,4}"
    r"(?P<name>[一-龥A-Za-z0-9·]{2,8})"
    r"(?:[^「」\n]{0,12}?)"
    r"(?:说道|说|道|问道|问|答道|答|喊道|喊|说着|如此说)"
)


# 受话介词：其后的角色名是受话对象而非说话人（"我对着小香穗说道"→香穗是受话人）。
_ADDRESSEE_PREP = _re.compile(r"(?:对着?|朝着?|冲|向|望着?|看向|注视着?|盯着?|看着)")


def _role_is_addressee(clause: str, role: str) -> bool:
    """该角色名在子句中是否为受话对象——其最近出现位置之前存在受话介词。
    用于让保守显性归因在受话人陷阱时放弃（交给 LLM/叙述者锚点），不做高置信误判。"""
    aliases = [role]
    if len(role) >= 4:
        aliases.append(role[-3:])
    if len(role) >= 3:
        aliases.append(role[-2:])
    rpos = max((clause.rfind(a) for a in aliases), default=-1)
    if rpos < 0:
        return False
    return any(m.end() <= rpos for m in _ADDRESSEE_PREP.finditer(clause[:rpos + 1]))


def _find_role_in_clause(clause: str, role_hints: list[str]) -> str | None:
    """
    在文本片段中查找最靠近末尾的 role_hint（或其简称）出现位置。
    返回完整的角色名（canonical），找不到返回 None。

    支持简称匹配：role_hints 中"王冢真唯"可以匹配文本中的"真唯"。
    """
    # 构建所有可能的简称：每个角色名的后 2、3 字
    candidates: list[tuple[str, str]] = []  # (搜索串, 完整名)
    for role in role_hints:
        candidates.append((role, role))          # 全名
        if len(role) >= 3:
            candidates.append((role[-2:], role)) # 后2字简称
        if len(role) >= 4:
            candidates.append((role[-3:], role)) # 后3字简称

    # 找最靠近末尾的匹配
    best_pos = -1
    best_role = None
    for search, full_role in candidates:
        pos = clause.rfind(search)
        if pos > best_pos:
            best_pos = pos
            best_role = full_role

    return best_role


def attribute_explicit_conservative(
    quote: QuoteSpan,
    role_hints: list[str],
) -> Attribution | None:
    """
    保守版显性归因：

    逻辑：
    1. 前显性：前文末尾句有说话动词 AND 前文中出现了已知角色名 → 归因给该角色
    2. 后显性：后文前 40 字有 [角色名简称][说话动词] → 归因给该角色

    与原 rule_attributor 的区别：
    - 支持日式长名的简称匹配（"真唯"→"王冢真唯"）
    - 名字必须在 role_hints 中，不接受陌生名字
    - 无法确认时返回 None，由 LLM 层接管（精确率优先于召回率）
    """
    if not role_hints:
        return None

    # 取前文最后一句（句号/换行之后的部分）
    cb_full = (quote.context_before or "")
    # 最后 120 字，再取最后一个句子
    cb_window = cb_full[-120:]
    last_sent_parts = _re.split(r"[。！？\n]", cb_window)
    cb_clause = last_sent_parts[-1].strip() if last_sent_parts else cb_window

    ca = (quote.context_after or "")[:60]

    # ── 前显性 ──
    if _SPEECH_VERB_END.search(cb_clause) and len(cb_clause) >= 4:
        # 只在最后逗号之后的子句里找角色名，避免假设/条件从句（"若是X说的"）误匹配
        tight_parts = _re.split(r"[，、]", cb_clause)
        tight_clause = tight_parts[-1].strip() if tight_parts else cb_clause
        role = _find_role_in_clause(tight_clause, role_hints)
        if role and not _role_is_addressee(tight_clause, role):
            evidence_text = cb_clause[-40:].strip()
            return Attribution(
                quote_id=quote.quote_id,
                speaker=role,
                confidence=0.93,
                evidence=f"前显性：「{evidence_text}」",
                attribution_type=AttributionType.EXPLICIT_BEFORE,
            )

    # ── 后显性 ──
    m = _AFTER_NAME_VERB.search(ca[:40])
    if m:
        raw_name = m.group("name").strip()
        # 找到完整角色名
        role = None
        for r in role_hints:
            if r == raw_name or r.endswith(raw_name) or r.startswith(raw_name):
                role = r
                break
        if role:
            return Attribution(
                quote_id=quote.quote_id,
                speaker=role,
                confidence=0.93,
                evidence=f"后显性：「{ca[:30].strip()}」",
                attribution_type=AttributionType.EXPLICIT_AFTER,
            )

    return None
