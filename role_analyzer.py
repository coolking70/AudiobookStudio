import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from typing import Any

from character_registry import CharacterRegistry
from llm_client import OpenAICompatibleClient
from schemas import AnalysisContext, LLMConfig

VALID_INSTRUCT_ITEMS = {
    "american accent",
    "australian accent",
    "british accent",
    "canadian accent",
    "child",
    "chinese accent",
    "elderly",
    "female",
    "high pitch",
    "indian accent",
    "japanese accent",
    "korean accent",
    "low pitch",
    "male",
    "middle-aged",
    "moderate pitch",
    "portuguese accent",
    "russian accent",
    "teenager",
    "very high pitch",
    "very low pitch",
    "whisper",
    "young adult",
}

VALID_EMOTIONS = {"neutral", "soft", "cold", "serious", "curious", "angry"}
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.2
DEFAULT_CHUNK_MAX_CHARS = 6000
SEED_CHUNK_MAX_CHARS = 12000
SAMPLE_DETECT_MAX_CHARS = 5000
SUPPORTED_NON_VERBAL_TAGS = [
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
]

DEFAULT_SYSTEM_PROMPT = """
你是一个”小说/对白文本分析器”。
请把用户提供的原文分析成适合 TTS 的 JSON 结构。

要求：
1. 输出必须是合法 JSON 对象。
2. 格式固定为：
{
  “segments”: [
    {
      “speaker”: “旁白”,
      “text”: “夜色降临。”,
      “emotion”: “neutral”,
      “style”: “female, moderate pitch”
    }
  ]
}
3. speaker 只填角色名、”旁白”或”UNKNOWN”（见规则 7）。
4. emotion 只能从以下值中选：neutral, soft, cold, serious, curious, angry
5. style 必须只使用以下英文标签，且用英文逗号+空格分隔：
american accent, australian accent, british accent, canadian accent,
child, chinese accent, elderly, female, high pitch, indian accent,
japanese accent, korean accent, low pitch, male, middle-aged,
moderate pitch, portuguese accent, russian accent, teenager,
very high pitch, very low pitch, whisper, young adult
6. 不要输出任何解释、注释、markdown，只输出 JSON。
7. 当无法可靠判断说话人时，speaker 填 “UNKNOWN”，不要硬猜成已知角色名；程序会将 UNKNOWN 自动归入旁白，避免错误归属。
8. 切分粒度：以”一段旁白叙述”或”一轮对白”为基本单位。相邻旁白叙述可合并为 1-3 句一段，但绝不允许省略或丢弃任何旁白内容。
9. 对白与旁白必须分属不同的 segment。引号内的对白归角色，引号外的叙述（含动作、场景、心理描写）归”旁白”。即使它们在同一个自然句中（如：”你好。”他说。），也必须拆成两个 segment：一个角色对白、一个旁白叙述。
10. 旁白段只包含叙述，不包含引号对白；角色段只包含该角色说的对白，不包含叙述动作。例如 “李明摇了摇头：”不行。””应拆为：旁白”李明摇了摇头：”+ 李明”不行。”。
11. 只有当单段明显过长时，才继续往下细分。
12. 根据角色的性别、年龄、性格特征来选择合适的 style 标签组合。
13. 根据对白的语境和情绪来选择合适的 emotion。
14. **严禁省略任何原文内容**：所有 segments 的 text 按顺序拼接起来必须覆盖原文的每一个字。
15. 判断 speaker 时务必结合上下文语义：
    a. “A 向/对 B 发问/说道/回答”之后的对白通常是 A 说的，不是 B；
    b. 内容为”是的/不是/没错/我不知道/嗯”等答复时，说话人优先判断为被提问的 B，而非提问的 A；
    c. 对白中出现”X 大人/殿下/阁下/您”等敬称时，说话人通常不是被称呼的 X 本人；
    d. 不要简单沿用上一轮对白的说话人——每句都要重新根据上下文判断。
""".strip()

# 上下文感知模式追加到 system prompt 的额外规则（context_mode != “off” 时注入）
_CONTEXT_RULES_ADDON = """
16. 本次分析的 user 消息中附带了【跨块分析上下文】，请务必结合这些信息判断 speaker：
    - “已知角色”是全书迄今出现过的角色名，speaker 应优先从中选取；
    - “上一块剧情摘要”描述紧接本块之前的情节，用于判断谁在说话；
    - “上一块仍在对话的角色”是对话场景延续的提示。
17. 对真正无法判断的对白，请填 “UNKNOWN”，切勿把猜测当事实。
18. 除 segments 之外，输出 JSON 还需包含（供下一块使用，程序自动删除）：
    “context_summary”: “一句话概括本块结尾的剧情/对话状态”,
    “active_speakers”: [“本块结尾仍活跃在对话中的角色名”],
    “new_characters”: [“本块中首次登场的新角色名，无则填 []”]
""".strip()

SPEAKER_VERIFICATION_PROMPT = """
你是”TTS 对白说话人核查专员”。你的唯一任务是核查给定 JSON segments 中的 speaker 字段是否与原文上下文一致。

规则：
1. 只检查和修正 speaker，绝对不修改 text、emotion、style 字段。
2. 只在高置信度时才输出修正建议；有疑问则跳过，绝不乱猜。
3. 常见错误场景（优先排查）：
   - “A 向 B 发问”后的对白被标为 B，实应为 A；
   - 回答内容（是的/不是/没错）被标为提问方，实应为被提问方；
   - 对白中称呼”X 大人/殿下”，被错标为 X 本人；
   - 连续轮次中错把上一段的说话人沿用到下一段。
4. 对每个有问题的 segment，输出其 index（从 0 起算）和建议 speaker。
5. 如果所有 speaker 都正确，输出空 corrections 列表。

只输出合法 JSON：
{
  “corrections”: [
    {“index”: 3, “suggested_speaker”: “加雷宁”, “confidence”: “high”, “reason”: “此段是对米拉提问的回答”}
  ]
}
""".strip()

CHARACTER_ALIAS_RESOLUTION_PROMPT = """
你是”人物别名归并专员”。输入是从小说中识别出的所有说话人名列表，里面可能存在同一个角色的多种叫法（全名、简称、敬称、绰号）。

请整理成别名分组，只有非常确定两个名字指的是同一人时才归并，不确定时保持独立。

只输出合法 JSON：
{
  “groups”: [
    {“canonical”: “规范名”, “aliases”: [“别名1”, “别名2”]}
  ]
}

如果没有可以合并的别名，输出 {“groups”: []}。
不要输出解释或 markdown。
""".strip()

SEGMENT_PLAN_PROMPT = """
你是一个“长文本 TTS 智能分段规划器”。
目标是把输入文本拆成适合后续角色分析的多个工作块，以便并发调用 LLM。

要求：
1. 只输出合法 JSON，对象格式固定为：
{
  "chunks": [
    {
      "title": "第1块",
      "content": "原文片段"
    }
  ]
}
2. content 必须直接来自用户原文，只允许做极少量空白整理，不要改写内容，不要补写，**严禁删减任何句子**——包括旁白叙述、动作描写、场景描写、过渡语句等，所有 chunk 的 content 拼接起来必须等于完整原文。
3. 优先按章节、场景切换、自然段和完整对白回合切分。
4. 每个 chunk 尽量语义完整，不要把一句对白拆开。
5. 如果用户原文里有章节标题，优先保留到对应 chunk 的 title。
6. 如果当前文本本来就很短，可以只返回 1 个 chunk。
7. 不要输出解释、注释或 markdown。
""".strip()

CHAPTER_RULE_DETECTION_PROMPT = """
你是一个“长文本章节规则探测器”。
请根据给出的文本样本，判断这篇文本最像哪一种章节标题模式。

只允许从以下 pattern 中选择一个：
- md_chapter_cn: Markdown 标题形式的“第X章”
- plain_chapter_cn: 纯文本形式的“第X章”
- cn_hui_jie: “第X回 / 第X节 / 第X幕 / 第X卷 / 第X篇”
- chapter_en: “Chapter 1”
- part_en: “Part 1”
- numbered: “1. 标题 / 1、标题 / 1）标题”
- md_heading: 通用 Markdown h1/h2
- none: 样本看不出稳定章节标题规则

输出只允许是合法 JSON，格式固定为：
{
  "pattern": "none",
  "confidence": "low",
  "reason": "一句简短判断理由"
}
""".strip()

_CN_NUM = r"[零一二三四五六七八九十百千万\d]+"
HEADING_PATTERNS: list[tuple[re.Pattern[str], str, int]] = [
    (re.compile(r"^#{1,3}\s*第\s*" + _CN_NUM + r"\s*章.*$", re.M), "md_chapter_cn", 10),
    (re.compile(r"^第\s*" + _CN_NUM + r"\s*章[：:．.\s].*$", re.M), "plain_chapter_cn", 8),
    (re.compile(r"^#{0,3}\s*第\s*" + _CN_NUM + r"\s*[回节幕卷篇].*$", re.M), "cn_hui_jie", 7),
    (re.compile(r"^#{0,3}\s*[Cc]hapter\s+\d+.*$", re.M), "chapter_en", 7),
    (re.compile(r"^#{0,3}\s*[Pp]art\s+\d+.*$", re.M), "part_en", 5),
    (re.compile(r"^#{0,3}\s*\d+[.、）)\s]+\S.*$", re.M), "numbered", 3),
    (re.compile(r"^#{1,2}\s+\S.*$", re.M), "md_heading", 2),
]
HEADING_PATTERN_MAP = {desc: regex for regex, desc, _ in HEADING_PATTERNS}
SCENE_BREAK_RE = re.compile(r"^[\s]*([*]{3,}|[-]{3,}|[=]{3,}|[—]{3,})[\s]*$", re.M)

TEXT_OPTIMIZATION_PROMPT = f"""
你是一个“OmniVoice 文本语音适配优化器”。
你的任务是在不改变原意、剧情、角色归属的前提下，把输入文本整理成更适合 OmniVoice 合成的版本。

请参考 OmniVoice 官方能力：
1. 支持内联非语言标签：{", ".join(SUPPORTED_NON_VERBAL_TAGS)}
2. 中文发音纠正可用带声调数字的拼音，如 ZHE2、SHE2、ZHE1。

优化原则：
1. **严禁删减原文内容**：所有旁白叙述（动作描写、场景描写、外貌描写、心理描写、过渡语句）必须完整保留，不得省略。优化后的 content 在去掉 TTS 标签后，必须包含原文的每一句话。
2. 尽量保持原句顺序、段落结构和含义不变。
3. 识别明显的语气词、叹息、笑声、疑问尾音，可在必要时替换成 OmniVoice 支持的标签，但被替换的叙述原文（如"他叹了口气"）仍需保留在旁白中，标签只是额外插入的辅助。
4. 识别多音字、容易误读的人名地名、生僻字，可在必要时改写为 OmniVoice 可读的拼音形式（使用带声调数字的拼音）。
5. 将 OmniVoice 可能误解的特殊符号做保守替换，例如把【】替换成更自然的中文表达或标点。
6. 在容易导致连读的空白换行之间补入必要标点，但不要过度加标点。
7. 不要随意改写文学风格；如果没有必要优化，就尽量少改。
7. 输出只允许是合法 JSON，格式固定为：
{{
  "chunks": [
    {{
      "title": "原title",
      "content": "优化后的文本"
    }}
  ]
}}
8. 不要输出解释、注释或 markdown。
""".strip()

SEGMENT_OPTIMIZE_PROMPT = f"""
你是一个“长文本 TTS 分段与语音适配一体化规划器”。
你的任务是把输入文本直接拆成适合后续角色分析的工作块，并同时完成 OmniVoice 友好的文本优化。

要求：
1. 只输出合法 JSON，对象格式固定为：
{{
  "chunks": [
    {{
      "title": "第1块",
      "content": "已经完成语音适配优化的文本片段"
    }}
  ]
}}
2. content 必须完整覆盖原文，不得删减任何句子、对白、动作描写、场景描写、心理描写或过渡语句。
3. 优先按章节、场景切换、自然段和完整对白回合切分，每个 chunk 尽量语义完整。
4. 在不改变剧情、语义和角色归属的前提下，把文本整理成更适合 OmniVoice 合成的版本；可参考以下优化原则：
   - 支持内联非语言标签：{", ".join(SUPPORTED_NON_VERBAL_TAGS)}
   - 中文发音纠正可用带声调数字的拼音，如 ZHE2、SHE2、ZHE1
   - 只做必要优化，不要过度改写文学风格
5. 所有 chunk 的 content 拼接起来必须仍能完整覆盖原文内容；不要输出解释、注释或 markdown。
""".strip()

OPTIMIZE_ANALYZE_PROMPT = """
你是一个“TTS 语音适配与角色分析一体化处理器”。
请在一次处理里同时完成两件事：
1. 在不改变原意、剧情和角色归属的前提下，把文本整理成更适合 OmniVoice 合成的版本；
2. 把整理后的文本分析成适合 TTS 的 JSON segments。

要求：
1. 输出必须是合法 JSON，对象格式固定为：
{
  "segments": [
    {
      "speaker": "旁白",
      "text": "优化后的文本片段",
      "emotion": "neutral",
      "style": "female, moderate pitch"
    }
  ]
}
2. 所有 segments 的 text 按顺序拼接后，必须完整覆盖原文内容；严禁删减任何句子。
3. speaker 只填角色名或“旁白”；emotion 只能从以下值中选：neutral, soft, cold, serious, curious, angry。
4. style 只能使用系统允许的英文标签，并用英文逗号加空格分隔。
5. 旁白与对白必须严格拆开；引号外叙述归“旁白”，引号内对白归角色。
6. text 字段允许包含为了合成稳定性所做的必要保守优化，但不要过度润色或改写。
7. 不要输出解释、注释、markdown 或额外说明。
""".strip()


def normalize_style(style: str | None) -> str | None:
    if not style:
        return None
    parts = [x.strip().lower() for x in str(style).split(",") if x.strip()]
    parts = [x for x in parts if x in VALID_INSTRUCT_ITEMS]
    if not parts:
        return None
    dedup = []
    for item in parts:
        if item not in dedup:
            dedup.append(item)
    return ", ".join(dedup)


def fallback_style_by_role(speaker: str, emotion: str) -> str:
    if speaker == "旁白":
        return "female, moderate pitch"
    if emotion in {"cold", "serious", "angry"}:
        return "male, low pitch"
    return "male, young adult, moderate pitch"


def sanitize_segments(
    segments: list[dict[str, Any]],
    known_characters: list[str] | None = None,
) -> list[dict[str, Any]]:
    """校验并清理 LLM 返回的 segments。

    known_characters: 可选的已知角色名集合，用于标记 _suspicious 字段。
    """
    known_set: set[str] = set(known_characters) if known_characters else set()
    cleaned = []
    for seg in segments:
        raw_speaker = str(seg.get("speaker", "旁白")).strip() or "旁白"
        text = str(seg.get("text", "")).strip()
        emotion = str(seg.get("emotion", "neutral")).strip().lower() or "neutral"
        style = normalize_style(seg.get("style"))

        if not text:
            continue
        if emotion not in VALID_EMOTIONS:
            emotion = "neutral"

        # UNKNOWN → 旁白，但记录标志供前端展示
        was_unknown = raw_speaker == "UNKNOWN"
        speaker = "旁白" if was_unknown else raw_speaker

        if not style:
            style = fallback_style_by_role(speaker, emotion)

        entry: dict[str, Any] = {
            "speaker": speaker,
            "text": text,
            "emotion": emotion,
            "style": style,
            "ref_audio": None,
            "ref_text": None,
        }
        if was_unknown:
            entry["_was_unknown"] = True
        # 标记可疑 speaker：不在已知角色中且不是旁白
        if known_set and speaker != "旁白" and speaker not in known_set:
            entry["_suspicious"] = True

        cleaned.append(entry)
    return cleaned


def detect_heading_pattern_by_rules(text: str) -> tuple[re.Pattern[str] | None, str, int]:
    best_pattern = None
    best_desc = "none"
    best_score = 0
    best_count = 0

    for regex, desc, weight in HEADING_PATTERNS:
        count = len(regex.findall(text))
        if count < 2:
            continue
        score = count * weight
        if score > best_score:
            best_score = score
            best_count = count
            best_pattern = regex
            best_desc = desc

    return best_pattern, best_desc, best_count


def split_by_detected_headings(text: str, pattern: re.Pattern[str]) -> list[dict[str, str]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    chunks = []
    for index, match in enumerate(matches):
        title = match.group(0).strip().lstrip("#").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            chunks.append({"title": title, "content": content})
    return chunks


def split_by_scene_breaks(text: str, title_prefix: str = "段落") -> list[dict[str, str]]:
    parts = re.split(SCENE_BREAK_RE, text)
    chunks = []
    for part in parts:
        content = str(part or "").strip()
        if content and not SCENE_BREAK_RE.fullmatch(content):
            chunks.append({"title": f"{title_prefix} {len(chunks) + 1}", "content": content})
    return chunks if len(chunks) >= 2 else []


def split_by_blank_lines(text: str, target_chars: int, max_chars: int, title_prefix: str = "块") -> list[dict[str, str]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        normalized = str(text or "").strip()
        return [{"title": f"{title_prefix} 1", "content": normalized}] if normalized else []

    chunks: list[dict[str, str]] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append({"title": f"{title_prefix} {len(chunks) + 1}", "content": "\n\n".join(current)})
            current = []
            current_len = 0

    for para in paragraphs:
        plen = len(para)
        next_len = current_len + plen + (2 if current else 0)
        if current and next_len > max_chars:
            flush()

        current.append(para)
        current_len += plen + (2 if len(current) > 1 else 0)

        if current_len >= target_chars:
            is_dialogue_end = para.rstrip().endswith(("”", "\"", "」", "』"))
            is_narration = not para.lstrip().startswith(("“", "\"", "「", "『"))
            if is_dialogue_end or is_narration:
                flush()

    flush()
    return chunks


def split_by_chapters(text: str) -> list[dict[str, str]]:
    pattern, _, _ = detect_heading_pattern_by_rules(text)
    if pattern:
        chunks = split_by_detected_headings(text, pattern)
        if chunks:
            return chunks

    normalized = text.strip()
    return [{"title": "全文", "content": normalized}] if normalized else []


def split_chunk_if_too_long(chunk: dict[str, str], max_chars: int) -> list[dict[str, str]]:
    content = chunk["content"]
    if len(content) <= max_chars:
        return [chunk]

    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", content) if item.strip()]
    if not paragraphs:
        return [chunk]

    sub_chunks = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > max_chars:
            sub_chunks.append(
                {
                    "title": f"{chunk['title']}（第{len(sub_chunks) + 1}部分）",
                    "content": "\n\n".join(current),
                }
            )
            current = [paragraph]
            current_len = len(paragraph)
            continue
        current.append(paragraph)
        current_len += len(paragraph)

    if current:
        sub_chunks.append(
            {
                "title": f"{chunk['title']}（第{len(sub_chunks) + 1}部分）",
                "content": "\n\n".join(current),
            }
        )

    return sub_chunks or [chunk]


def _extract_detection_sample(text: str, limit: int = SAMPLE_DETECT_MAX_CHARS) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized

    lines = [line.rstrip() for line in normalized.splitlines() if line.strip()]
    sample_lines: list[str] = []
    sample_len = 0
    for line in lines:
        sample_lines.append(line)
        sample_len += len(line) + 1
        if sample_len >= limit:
            break
    return "\n".join(sample_lines).strip()[:limit]


def resolve_chapter_sample_limit(llm_config: LLMConfig | None = None) -> int:
    configured = int(getattr(llm_config, "chapter_sample_chars", 0) or 0) if llm_config is not None else 0
    return max(600, min(configured or SAMPLE_DETECT_MAX_CHARS, 20000))


def infer_heading_pattern_with_llm(sample_text: str, llm_config: LLMConfig) -> dict[str, str]:
    chapter_prompt = llm_config.chapter_prompt.strip() if llm_config.chapter_prompt else CHAPTER_RULE_DETECTION_PROMPT
    messages = [
        {"role": "system", "content": chapter_prompt},
        {
            "role": "user",
            "content": f"请判断下面文本样本最像哪一种章节标题模式。\n\n文本样本：\n{sample_text}",
        },
    ]
    result = _run_chat_json_with_retry(llm_config, messages, "章节规则探测失败", task_kind="segment")
    return {
        "pattern": str(result.get("pattern") or "none").strip(),
        "confidence": str(result.get("confidence") or "").strip(),
        "reason": str(result.get("reason") or "").strip(),
    }


def _build_segmentation_params(llm_config: LLMConfig) -> tuple[int, int]:
    runtime = str(getattr(llm_config, "local_runtime", "") or "").strip().lower()
    engine = str(getattr(llm_config, "local_engine", "") or "").strip().lower()
    model_path = str(getattr(llm_config, "local_model_path", "") or getattr(llm_config, "model", "") or "").strip().lower()
    is_direct_gguf = runtime == "direct" and (engine == "gguf" or model_path.endswith(".gguf"))
    if is_direct_gguf:
        return 4200, 5600
    if runtime == "direct":
        return 5200, 7600
    return DEFAULT_CHUNK_MAX_CHARS, SEED_CHUNK_MAX_CHARS


def detect_chapter_structure_with_info(text: str, llm_config: LLMConfig | None = None, max_chars: int = SEED_CHUNK_MAX_CHARS) -> tuple[list[dict[str, str]], dict[str, Any]]:
    normalized = str(text or "").strip()
    info: dict[str, Any] = {
        "method": "empty",
        "heading_pattern": "none",
        "resolved_heading_regex": "",
        "chapter_count": 0,
        "seed_chunk_count": 0,
        "used_llm_detection": False,
        "llm_pattern": "none",
        "llm_reason": "",
        "chapter_sample_chars": resolve_chapter_sample_limit(llm_config),
    }
    if not normalized:
        return [], info

    target_chars, resolved_max_chars = _build_segmentation_params(llm_config or LLMConfig(base_url="", api_key="", model="")) if llm_config else (DEFAULT_CHUNK_MAX_CHARS, max_chars)
    resolved_max_chars = max(resolved_max_chars, target_chars)

    pattern = None
    desc = "none"
    chapter_count = 0
    regex_override = str(getattr(llm_config, "chapter_regex_override", "") or "").strip() if llm_config is not None else ""
    if regex_override:
        try:
            override_pattern = re.compile(regex_override, re.M)
            override_count = len(list(override_pattern.finditer(normalized)))
            info["resolved_heading_regex"] = regex_override
            if override_count >= 1:
                pattern = override_pattern
                desc = "custom_regex"
                chapter_count = override_count
        except re.error as exc:
            info["llm_reason"] = f"自定义章节正则无效，已回退自动识别：{exc}"

    if pattern is None:
        pattern, desc, chapter_count = detect_heading_pattern_by_rules(normalized)
        if pattern is not None:
            info["resolved_heading_regex"] = pattern.pattern

    if pattern is None and llm_config is not None:
        sample_limit = resolve_chapter_sample_limit(llm_config)
        sample = _extract_detection_sample(normalized, limit=sample_limit)
        try:
            detection = infer_heading_pattern_with_llm(sample, llm_config)
            info["used_llm_detection"] = True
            info["llm_pattern"] = detection["pattern"]
            info["llm_reason"] = detection["reason"]
            guessed_pattern = HEADING_PATTERN_MAP.get(detection["pattern"])
            if guessed_pattern is not None:
                guessed_count = len(list(guessed_pattern.finditer(normalized)))
                if guessed_count >= 2:
                    pattern = guessed_pattern
                    desc = detection["pattern"]
                    chapter_count = guessed_count
                    info["resolved_heading_regex"] = guessed_pattern.pattern
        except Exception as exc:
            info["used_llm_detection"] = True
            info["llm_pattern"] = "none"
            info["llm_reason"] = f"探测失败，已回退到纯规则切块：{exc}"

    chunks: list[dict[str, str]] = []
    if pattern is not None:
        chapters = split_by_detected_headings(normalized, pattern)
        if chapters:
            info["method"] = "heading"
            info["heading_pattern"] = desc
            info["chapter_count"] = len(chapters)
            for chapter in chapters:
                if len(chapter["content"]) <= resolved_max_chars:
                    chunks.append(chapter)
                    continue
                scene_chunks = split_by_scene_breaks(chapter["content"], chapter["title"])
                if scene_chunks:
                    for scene_chunk in scene_chunks:
                        if len(scene_chunk["content"]) <= resolved_max_chars:
                            chunks.append(scene_chunk)
                        else:
                            chunks.extend(split_by_blank_lines(scene_chunk["content"], target_chars, resolved_max_chars, scene_chunk["title"]))
                else:
                    chunks.extend(split_by_blank_lines(chapter["content"], target_chars, resolved_max_chars, chapter["title"]))

    if not chunks:
        scene_chunks = split_by_scene_breaks(normalized)
        if scene_chunks:
            info["method"] = "scene_break"
            chunks = []
            for scene_chunk in scene_chunks:
                if len(scene_chunk["content"]) <= resolved_max_chars:
                    chunks.append(scene_chunk)
                else:
                    chunks.extend(split_by_blank_lines(scene_chunk["content"], target_chars, resolved_max_chars, scene_chunk["title"]))
        else:
            info["method"] = "paragraph"
            chunks = split_by_blank_lines(normalized, target_chars, resolved_max_chars, "全文")

    if not chunks:
        chunks = [{"title": "全文", "content": normalized}]
        info["method"] = "fallback_single"

    info["seed_chunk_count"] = len(chunks)
    info["target_chars"] = target_chars
    info["max_chars"] = resolved_max_chars
    attach_line_index_to_chunks(chunks)
    return chunks, info


def build_seed_chunks(text: str, llm_config: LLMConfig | None = None, max_chars: int = SEED_CHUNK_MAX_CHARS) -> tuple[list[dict[str, str]], dict[str, Any]]:
    return detect_chapter_structure_with_info(text, llm_config=llm_config, max_chars=max_chars)


# ── 预处理：行号化 + 软切 ──────────────────────────────────────────────────
# 把原文按"自然换行 + 句末标点软切"切成行，每行带绝对偏移量。
# 这是 compact-output 模式的基础：LLM 用 L1/L2 行号 + 引号锚点定位段落，
# 程序按 lines 索引精确还原原文 span，避免 LLM 数字符出错。

# 句末终结标点（仅在这些位置软切；逗号/分号/引号内绝不切）
# 句末标点之后允许跟随的"收尾符"：全角/半角引号、圆括号——保证不会把闭合引号孤立到下一行
_SENTENCE_END_RE = re.compile(r"[。！？!?…]+[”』」'\")）]*")


def _soft_split_paragraph(text: str, base_offset: int, max_line_chars: int) -> list[dict[str, Any]]:
    """把一个长段落按句末标点切成多个"句行"。

    返回每行的 {start, end, text}，偏移量是相对原文的绝对位置。
    短段落（<= max_line_chars）原样返回单行。
    """
    n = len(text)
    if n <= max_line_chars:
        return [{"start": base_offset, "end": base_offset + n, "text": text}]

    splits: list[int] = []  # 候选切点（绝对偏移，切点之前是句末）
    for m in _SENTENCE_END_RE.finditer(text):
        splits.append(m.end())

    if not splits or splits[-1] != n:
        splits.append(n)

    lines: list[dict[str, Any]] = []
    seg_start = 0
    cur_start = 0
    for sp in splits:
        # 当前累积长度超过阈值时切一段
        if sp - cur_start >= max_line_chars and seg_start < sp:
            lines.append({
                "start": base_offset + cur_start,
                "end": base_offset + sp,
                "text": text[cur_start:sp],
            })
            cur_start = sp
        seg_start = sp

    # 收尾
    if cur_start < n:
        lines.append({
            "start": base_offset + cur_start,
            "end": base_offset + n,
            "text": text[cur_start:n],
        })
    return lines or [{"start": base_offset, "end": base_offset + n, "text": text}]


def number_lines_with_soft_split(content: str, max_line_chars: int = 160) -> dict[str, Any]:
    """把一个 chunk 的纯文本预处理成行号化结构。

    返回:
        {
            "numbered_text": "L1: ...\\nL2: ...",  # 喂给 LLM 的版本
            "lines": [{"id": "L1", "start": 0, "end": 23, "text": "..."}],
        }

    保证：lines 的 text 拼接后等价于剔除空行后的原文（保留语义完整）。
    """
    lines: list[dict[str, Any]] = []
    raw = str(content or "")
    if not raw.strip():
        return {"numbered_text": "", "lines": []}

    # 按自然换行切成段（保留每段的绝对起始偏移）
    cursor = 0
    for raw_line in raw.split("\n"):
        line_len = len(raw_line)
        stripped = raw_line.strip()
        if stripped:
            # 计算 stripped 在原文里的绝对起始偏移（去掉左侧空白）
            left_skip = len(raw_line) - len(raw_line.lstrip())
            base = cursor + left_skip
            for piece in _soft_split_paragraph(stripped, base, max_line_chars):
                lines.append(piece)
        cursor += line_len + 1  # +1 for \n

    # 编号
    for idx, line in enumerate(lines, start=1):
        line["id"] = f"L{idx}"

    numbered_text = "\n".join(f"{ln['id']}: {ln['text']}" for ln in lines)
    return {"numbered_text": numbered_text, "lines": lines}


def attach_line_index_to_chunks(
    chunks: list[dict[str, Any]], max_line_chars: int = 160
) -> list[dict[str, Any]]:
    """给已切分好的 chunks 列表附加 numbered_text 和 lines 索引。

    不修改原 content，只新增字段。下游若不需要可忽略，向后兼容。
    """
    for chunk in chunks:
        if "lines" in chunk and "numbered_text" in chunk:
            continue  # 已处理过，跳过
        result = number_lines_with_soft_split(chunk.get("content", ""), max_line_chars)
        chunk["numbered_text"] = result["numbered_text"]
        chunk["lines"] = result["lines"]
    return chunks


def basic_tts_text_cleanup(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n")
    replacements = {
        "【": "“",
        "】": "”",
        "（": "，",
        "）": "，",
        "[": "【",
        "]": "】",
    }
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"([^\n。！？!?；;：:])\n([^\n])", r"\1。\n\2", cleaned)
    return cleaned.strip()


def _run_chat_json_with_retry(
    llm_config: LLMConfig,
    messages: list[dict[str, str]],
    error_prefix: str,
    task_kind: str = "general",
) -> dict[str, Any]:
    def is_direct_gguf_config(config: LLMConfig) -> bool:
        runtime = str(getattr(config, "local_runtime", "") or "").strip().lower()
        engine = str(getattr(config, "local_engine", "") or "").strip().lower()
        model_path = str(getattr(config, "local_model_path", "") or getattr(config, "model", "") or "").strip().lower()
        return runtime == "direct" and (engine == "gguf" or model_path.endswith(".gguf"))

    def build_retry_profiles(config: LLMConfig) -> list[LLMConfig]:
        profiles = [config]
        if not is_direct_gguf_config(config):
            return profiles

        base_ctx = int(config.local_ctx_tokens or 4096)
        base_batch = int(config.local_batch_size or 128)
        base_gpu_layers = config.local_gpu_layers
        if base_gpu_layers in (None, -1):
            base_gpu_layers = 16

        task_token_limits = {
            "segment": (384, 256),
            "optimize": (768, 512),
            "analyze": (1024, 768),
            "general": (1024, 768),
        }
        primary_token_limit, secondary_token_limit = task_token_limits.get(task_kind, task_token_limits["general"])
        primary_ctx = 2048 if task_kind == "segment" else 3072
        secondary_ctx = 1536 if task_kind == "segment" else 2048
        primary_batch = 32 if task_kind == "segment" else 64
        secondary_batch = 16 if task_kind == "segment" else 32
        primary_gpu_layers = 8 if task_kind == "segment" else 12
        secondary_gpu_layers = 4 if task_kind == "segment" else 8

        fallback_1 = config.model_copy(update={
            "workers": 1,
            "max_tokens": min(int(config.max_tokens or primary_token_limit), primary_token_limit),
            "local_ctx_tokens": min(base_ctx, primary_ctx),
            "local_batch_size": min(base_batch, primary_batch),
            "local_gpu_layers": min(int(base_gpu_layers), primary_gpu_layers),
        })
        fallback_2 = config.model_copy(update={
            "workers": 1,
            "max_tokens": min(int(config.max_tokens or secondary_token_limit), secondary_token_limit),
            "local_ctx_tokens": min(base_ctx, secondary_ctx),
            "local_batch_size": min(base_batch, secondary_batch),
            "local_gpu_layers": min(int(base_gpu_layers), secondary_gpu_layers),
        })
        profiles.extend([fallback_1, fallback_2])
        return profiles

    last_error: Exception | None = None
    retry_profiles = build_retry_profiles(llm_config)
    for profile_index, config in enumerate(retry_profiles, start=1):
        for attempt in range(1, MAX_RETRIES + 1):
            client = OpenAICompatibleClient(config)
            try:
                return client.chat_json(messages)
            except Exception as exc:
                last_error = exc
                if attempt >= MAX_RETRIES:
                    break
                time.sleep(RETRY_DELAY_SECONDS * attempt)
        if profile_index < len(retry_profiles):
            time.sleep(RETRY_DELAY_SECONDS)
    assert last_error is not None
    raise RuntimeError(f"{error_prefix}：{last_error}") from last_error


def _coerce_chunks(raw_chunks: list[dict[str, Any]], fallback_title: str) -> list[dict[str, str]]:
    normalized = []
    for index, chunk in enumerate(raw_chunks, start=1):
        title = str(chunk.get("title") or f"{fallback_title} - 第{index}块").strip() or f"{fallback_title} - 第{index}块"
        content = str(chunk.get("content") or "").strip()
        if content:
            normalized.append({"title": title, "content": content})
    return normalized


def intelligent_segment_seed_chunk(seed_chunk: dict[str, str], llm_config: LLMConfig) -> list[dict[str, str]]:
    char_budget = max(1800, min(DEFAULT_CHUNK_MAX_CHARS, int((llm_config.segment_target_chars or 0) or (llm_config.max_tokens * 2.4))))
    base_prompt = llm_config.segment_prompt.strip() if llm_config.segment_prompt else SEGMENT_PLAN_PROMPT
    prompt = (
        f"{base_prompt}\n\n"
        f"补充要求：单个 chunk 的 content 尽量控制在 {char_budget} 字以内；如果自然分段后某块略超出也可以接受，但不要过度细碎。"
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"请对下面文本做智能分段规划。\n\n标题：{seed_chunk['title']}\n\n正文：\n{seed_chunk['content']}",
        },
    ]
    result = _run_chat_json_with_retry(llm_config, messages, f"{seed_chunk['title']} 智能分段失败", task_kind="segment")
    chunks = _coerce_chunks(result.get("chunks", []), seed_chunk["title"])
    if chunks:
        return chunks
    return split_chunk_if_too_long(seed_chunk, max_chars=char_budget)


def segment_and_optimize_seed_chunk(seed_chunk: dict[str, str], llm_config: LLMConfig) -> list[dict[str, str]]:
    char_budget = max(1800, min(DEFAULT_CHUNK_MAX_CHARS, int((llm_config.segment_target_chars or 0) or (llm_config.max_tokens * 2.4))))
    base_prompt = llm_config.segment_optimize_prompt.strip() if llm_config.segment_optimize_prompt else SEGMENT_OPTIMIZE_PROMPT
    prompt = (
        f"{base_prompt}\n\n"
        f"补充要求：单个 chunk 的 content 尽量控制在 {char_budget} 字以内；如果自然分段后某块略超出也可以接受，但不要过度细碎。"
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"请对下面文本完成“智能分段 + 语音适配优化”。\n\n标题：{seed_chunk['title']}\n\n正文：\n{seed_chunk['content']}",
        },
    ]
    result = _run_chat_json_with_retry(llm_config, messages, f"{seed_chunk['title']} 分段与语音适配失败", task_kind="segment")
    chunks = _coerce_chunks(result.get("chunks", []), seed_chunk["title"])
    if chunks:
        return chunks
    fallback = split_chunk_if_too_long(seed_chunk, max_chars=char_budget)
    return [optimize_chunk_for_tts(chunk, llm_config) for chunk in fallback]


def intelligent_segment_text_with_info(text: str, llm_config: LLMConfig) -> tuple[list[dict[str, str]], dict[str, Any]]:
    seed_chunks, info = detect_chapter_structure_with_info(text, llm_config=llm_config)
    if not seed_chunks:
        return [], info

    direct_runtime = str(getattr(llm_config, "local_runtime", "") or "").strip().lower() == "direct"
    workers = 1 if direct_runtime else max(1, min(llm_config.workers or 5, len(seed_chunks)))
    results: list[list[dict[str, str]] | None] = [None] * len(seed_chunks)
    info["llm_workers"] = workers

    def _work(index: int, seed_chunk: dict[str, str]) -> tuple[int, list[dict[str, str]]]:
        return index, intelligent_segment_seed_chunk(seed_chunk, llm_config)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_work, idx, chunk) for idx, chunk in enumerate(seed_chunks)]
        for future in as_completed(futures):
            index, chunks = future.result()
            results[index] = chunks

    merged: list[dict[str, str]] = []
    for item in results:
        if item:
            merged.extend(item)
    info["planned_chunk_count"] = len(merged)
    return merged, info


def intelligent_segment_text(text: str, llm_config: LLMConfig) -> list[dict[str, str]]:
    chunks, _ = intelligent_segment_text_with_info(text, llm_config)
    return chunks


def segment_and_optimize_text_with_info(text: str, llm_config: LLMConfig) -> tuple[list[dict[str, str]], dict[str, Any]]:
    seed_chunks, info = detect_chapter_structure_with_info(text, llm_config=llm_config)
    if not seed_chunks:
        return [], info

    direct_runtime = str(getattr(llm_config, "local_runtime", "") or "").strip().lower() == "direct"
    workers = 1 if direct_runtime else max(1, min(llm_config.workers or 5, len(seed_chunks)))
    results: list[list[dict[str, str]] | None] = [None] * len(seed_chunks)
    info["llm_workers"] = workers
    info["combo_mode"] = "segment_optimize"

    def _work(index: int, seed_chunk: dict[str, str]) -> tuple[int, list[dict[str, str]]]:
        return index, segment_and_optimize_seed_chunk(seed_chunk, llm_config)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_work, idx, chunk) for idx, chunk in enumerate(seed_chunks)]
        for future in as_completed(futures):
            index, chunks = future.result()
            results[index] = chunks

    merged: list[dict[str, str]] = []
    for item in results:
        if item:
            merged.extend(item)
    info["planned_chunk_count"] = len(merged)
    info["optimized_chunk_count"] = len(merged)
    return merged, info


def optimize_chunk_for_tts(chunk: dict[str, str], llm_config: LLMConfig) -> dict[str, str]:
    cleaned = basic_tts_text_cleanup(chunk["content"])
    opt_prompt = llm_config.optimize_prompt.strip() if llm_config.optimize_prompt else TEXT_OPTIMIZATION_PROMPT
    messages = [
        {"role": "system", "content": opt_prompt},
        {
            "role": "user",
            "content": f"请优化下面这段文本，使其更适合 OmniVoice 合成。\n\n标题：{chunk['title']}\n\n正文：\n{cleaned}",
        },
    ]
    result = _run_chat_json_with_retry(llm_config, messages, f"{chunk['title']} 语音适配优化失败", task_kind="optimize")
    optimized_chunks = _coerce_chunks(result.get("chunks", []), chunk["title"])
    if optimized_chunks:
        return optimized_chunks[0]
    return {"title": chunk["title"], "content": cleaned}


def optimize_chunks_for_tts(chunks: list[dict[str, str]], llm_config: LLMConfig) -> list[dict[str, str]]:
    if not chunks:
        return []

    direct_runtime = str(getattr(llm_config, "local_runtime", "") or "").strip().lower() == "direct"
    workers = 1 if direct_runtime else max(1, min(llm_config.workers or 5, len(chunks)))
    results: list[dict[str, str] | None] = [None] * len(chunks)

    def _work(index: int, chunk: dict[str, str]) -> tuple[int, dict[str, str]]:
        return index, optimize_chunk_for_tts(chunk, llm_config)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_work, idx, chunk) for idx, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            index, optimized = future.result()
            results[index] = optimized

    return [item for item in results if item]


def _chapter_key(title: str) -> str:
    """从 chunk title 提取章节标识，用于 chapter 模式分组。

    例："第一章 - 第2块" → "第一章"；"第一章" → "第一章"；"块1" → "块1"
    """
    return (title or "").split(" - ")[0].strip()


# ── Compact 输出模式 ──────────────────────────────────────────────────────────
# 角色分析的紧凑输出格式，LLM 只输出行号表而非重复原文，节省 ~80% 输出 token。

COMPACT_ANALYSIS_PROMPT = """
你是小说文本角色分析器。输入是带行号的文本（L1: ... L2: ...）。

严格按以下格式输出，不输出任何 JSON、markdown 或解释文字：

ROLES: <编号>=<角色名>|<style标签>, ...
SEGS:
<行号> <编号> <情绪> [<锚点>]
...

---格式说明---

ROLES 行：列出本章节出现的非旁白角色，style 用空格分隔英文标签。
  示例：ROLES: 1=林烬|male young adult moderate pitch, 2=苏明|male low pitch
  0 固定代表"旁白"，无需声明。

SEGS 行：每行描述一个片段，顺序必须与原文一致。
  整行：  L3 1 cu         （L3 整行，角色=编号1，情绪=cu）
  前段：  L5 0 ne "他说。" （L5 从行首到"他说。"末尾，旁白）
  后段：  L5 1 se rest    （L5 接上一锚点之后到行末，角色=编号1）
  一行最多拆成两段（前段+rest）。

情绪代码：ne=neutral so=soft co=cold se=serious cu=curious an=angry
说话人：0=旁白，正整数=ROLES 中的角色编号，?=UNKNOWN

---要求---
- 所有行（L1, L2...）必须覆盖，不得遗漏
- 相邻旁白行可合并，如：L1-L3 0 ne
- 锚点文本必须与原文完全一致（用于程序精确定位）
- 仅当确有两个说话人在同一行时才使用锚点；不确定时整行归旁白
""".strip()

# 情绪代码 → 全称映射
_EMOTION_CODES: dict[str, str] = {
    "ne": "neutral", "so": "soft", "co": "cold",
    "se": "serious", "cu": "curious", "an": "angry",
    # 容错：允许全称
    "neutral": "neutral", "soft": "soft", "cold": "cold",
    "serious": "serious", "curious": "curious", "angry": "angry",
}

# 样式标签规范化：下划线 → 空格
def _norm_style(raw: str) -> str:
    return raw.replace("_", " ").strip()


def _decode_compact_output(
    text: str, chunk: dict[str, Any]
) -> list[dict[str, Any]]:
    """将 compact 格式的 LLM 输出解码为标准 segments 列表。

    chunk 必须已经过 attach_line_index_to_chunks（含 lines 字段）。
    遇到无法解析的行时，整行降级为旁白，不中断。
    """
    lines_index: list[dict[str, Any]] = chunk.get("lines", [])
    if not lines_index:
        raise ValueError("chunk 缺少 lines 索引，无法解码 compact 输出")

    line_by_id: dict[str, dict[str, Any]] = {ln["id"]: ln for ln in lines_index}
    content = chunk["content"]

    # ── 解析 ROLES ────────────────────────────────────────────────────────
    roles: dict[str, tuple[str, str]] = {"0": ("旁白", "female moderate pitch")}
    segs_text = text
    for block in text.split("\n"):
        block = block.strip()
        if block.startswith("ROLES:"):
            roles_raw = block[6:].strip()
            for entry in roles_raw.split(","):
                entry = entry.strip()
                if "=" not in entry:
                    continue
                rid, rest_r = entry.split("=", 1)
                rid = rid.strip()
                parts_r = rest_r.strip().split("|", 1)
                rname = parts_r[0].strip()
                rstyle = _norm_style(parts_r[1]) if len(parts_r) > 1 else ""
                if rid:
                    roles[rid] = (rname or "UNKNOWN", rstyle)
        if block.startswith("SEGS:"):
            segs_text = text[text.index("SEGS:") + 5:]
            break

    # ── 解析 SEGS ─────────────────────────────────────────────────────────
    segments: list[dict[str, Any]] = []
    # 记录每个 line_id 当前用完的字符偏移（相对 line.text）
    line_cursors: dict[str, int] = {}
    # 行号范围正则，如 L1-L3 或 L5
    _line_ref_re = re.compile(r"^(L\d+)(?:-(L\d+))?$")

    for raw_line in segs_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line or raw_line.startswith("SEGS:") or raw_line.startswith("ROLES:"):
            continue

        # 拆分为 token，保留引号内的锚点文本作为一个整体
        # 格式：line_ref  speaker  emotion  [anchor_or_rest]
        tokens = raw_line.split(None, 3)  # 最多 4 个 token
        if len(tokens) < 3:
            continue

        line_ref_token, speaker_token, emotion_token = tokens[0], tokens[1], tokens[2]
        anchor_token = tokens[3].strip() if len(tokens) > 3 else ""

        # 情绪
        emotion = _EMOTION_CODES.get(emotion_token.lower(), "neutral")

        # speaker
        speaker_token = speaker_token.strip()
        if speaker_token == "?":
            speaker = "UNKNOWN"
            style = ""
        elif speaker_token == "0":
            speaker = "旁白"
            style = roles.get("0", ("旁白", "female moderate pitch"))[1]
        else:
            role_info = roles.get(speaker_token)
            if role_info:
                speaker, style = role_info
            else:
                speaker, style = "UNKNOWN", ""

        # 展开行号范围（L1-L3 → [L1, L2, L3]）
        m = _line_ref_re.match(line_ref_token)
        if not m:
            continue
        start_id = m.group(1)
        end_id = m.group(2) or start_id
        # 找到范围内所有行
        ids_in_range: list[str] = []
        in_range = False
        for ln in lines_index:
            if ln["id"] == start_id:
                in_range = True
            if in_range:
                ids_in_range.append(ln["id"])
            if ln["id"] == end_id and in_range:
                break

        if not ids_in_range:
            continue

        for lid in ids_in_range:
            ln = line_by_id.get(lid)
            if not ln:
                continue
            line_text = ln["text"]
            cursor = line_cursors.get(lid, 0)

            is_last_lid = lid == ids_in_range[-1]

            if anchor_token == "rest" or anchor_token == "":
                # 整行（或从 cursor 到末尾）
                seg_text = line_text[cursor:]
                line_cursors[lid] = len(line_text)
            elif anchor_token.startswith('"') and anchor_token.endswith('"') and len(anchor_token) > 2:
                # 前段：从 cursor 到锚点末尾
                anchor = anchor_token[1:-1]
                idx = line_text.find(anchor, cursor)
                if idx == -1:
                    # 锚点找不到，降级整行
                    seg_text = line_text[cursor:]
                    line_cursors[lid] = len(line_text)
                else:
                    end_pos = idx + len(anchor)
                    seg_text = line_text[cursor:end_pos]
                    line_cursors[lid] = end_pos
            else:
                seg_text = line_text[cursor:]
                line_cursors[lid] = len(line_text)

            seg_text = seg_text.strip()
            if not seg_text:
                continue

            segments.append({
                "speaker": speaker,
                "text": seg_text,
                "emotion": emotion,
                "style": style,
            })

    if not segments:
        raise ValueError("compact 解码结果为空，可能格式有误")

    return segments


def _build_context_user_content(chunk: dict[str, str], context: AnalysisContext | None) -> str:
    """构建带上下文的 user 消息内容（verbose 模式）。"""
    parts: list[str] = []
    if context is not None and (
        context.known_characters or context.last_context_summary or context.last_active_speakers
    ):
        ctx_lines = ["【跨块分析上下文】"]
        if context.known_characters:
            ctx_lines.append(f"已知角色：{', '.join(context.known_characters)}")
        if context.last_context_summary:
            ctx_lines.append(f"上一块剧情摘要：{context.last_context_summary}")
        if context.last_active_speakers:
            ctx_lines.append(f"上一块仍在对话的角色：{', '.join(context.last_active_speakers)}")
        parts.append("\n".join(ctx_lines))

    parts.append(
        f"请分析下面这段文本，并输出 JSON。\n\n标题：{chunk['title']}\n\n正文：\n{chunk['content']}"
    )
    return "\n\n".join(parts)


def _build_compact_user_content(chunk: dict[str, Any], context: AnalysisContext | None) -> str:
    """构建带上下文的 user 消息内容（compact 模式）。
    发送的是 numbered_text（行号化文本），而非原始 content。
    """
    parts: list[str] = []
    if context is not None and (
        context.known_characters or context.last_context_summary or context.last_active_speakers
    ):
        ctx_lines = ["【跨块分析上下文】"]
        if context.known_characters:
            ctx_lines.append(f"已知角色：{', '.join(context.known_characters)}")
        if context.last_context_summary:
            ctx_lines.append(f"上一块剧情摘要：{context.last_context_summary}")
        if context.last_active_speakers:
            ctx_lines.append(f"上一块仍在对话的角色：{', '.join(context.last_active_speakers)}")
        parts.append("\n".join(ctx_lines))

    numbered = chunk.get("numbered_text") or chunk.get("content", "")
    parts.append(f"标题：{chunk['title']}\n\n{numbered}")
    return "\n\n".join(parts)


def _merge_known_characters(
    existing: list[str],
    new_speakers: list[str],
    extra: list[str] | None,
) -> list[str]:
    """合并已知角色列表，去重保序。"""
    seen: dict[str, None] = dict.fromkeys(existing)
    for name in (new_speakers or []) + (extra or []):
        n = (name or "").strip()
        if n and n not in ("旁白", "UNKNOWN", ""):
            seen[n] = None
    return list(seen.keys())


def _analyze_chunk_with_context(
    chunk: dict[str, str],
    llm_config: LLMConfig,
    context: AnalysisContext | None = None,
) -> tuple[list[dict[str, Any]], AnalysisContext]:
    """分析单个 chunk，可选择携带跨块上下文。

    返回 (segments, updated_context)。
    当 context=None 时行为与旧版 analyze_chunk_with_retry 完全一致。
    当 llm_config.output_mode == "compact" 时使用紧凑输出格式。
    """
    use_context = context is not None
    use_compact = str(getattr(llm_config, "output_mode", "verbose") or "verbose").lower() == "compact"

    # ── compact 模式 ──────────────────────────────────────────────────────
    if use_compact:
        if not chunk.get("lines"):
            # chunk 尚未行号化（如从旧快照恢复），先补充
            result = number_lines_with_soft_split(chunk.get("content", ""))
            chunk = {**chunk, "numbered_text": result["numbered_text"], "lines": result["lines"]}

        system_prompt = COMPACT_ANALYSIS_PROMPT
        if use_context and (context.known_characters or context.last_context_summary):
            ctx_addon = "\n\n【上下文】"
            if context.known_characters:
                ctx_addon += f"\n已知角色：{', '.join(context.known_characters)}"
            if context.last_context_summary:
                ctx_addon += f"\n上一块摘要：{context.last_context_summary}"
            if context.last_active_speakers:
                ctx_addon += f"\n上一块活跃角色：{', '.join(context.last_active_speakers)}"
            system_prompt = system_prompt + ctx_addon

        user_content = _build_compact_user_content(chunk, context if use_context else None)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        compact_text = ""
        try:
            client = OpenAICompatibleClient(llm_config)
            compact_text = client.chat_text(messages)
            segments = _decode_compact_output(compact_text, chunk)
        except Exception as exc:
            # compact 解码失败 → 回退到 verbose 模式
            import logging
            logging.warning(
                f"[compact] {chunk['title']} 解码失败（{exc}），回退 verbose 模式"
            )
            fallback_cfg = llm_config.model_copy(update={"output_mode": "verbose"})
            return _analyze_chunk_with_context(chunk, fallback_cfg, context)

        known_chars = context.known_characters if context else []
        segments = sanitize_segments(segments, known_characters=known_chars)

        # compact 模式：从解码结果推导 context，无需 LLM 额外输出
        new_speakers = [s["speaker"] for s in segments if s["speaker"] not in ("旁白", "UNKNOWN")]
        updated_known = _merge_known_characters(
            context.known_characters if context else [], new_speakers, None
        )
        # 取最后出现的活跃非旁白 speaker 作为 active_speakers
        active = list(dict.fromkeys(
            s["speaker"] for s in reversed(segments[-10:])
            if s["speaker"] not in ("旁白", "UNKNOWN")
        ))
        updated_context = AnalysisContext(
            known_characters=updated_known,
            character_aliases=context.character_aliases if context else {},
            last_context_summary="",  # compact 模式不输出摘要
            last_active_speakers=active[:3],
            chunk_index=(context.chunk_index + 1) if context else 1,
        )
        return segments, updated_context

    # ── verbose 模式（原有逻辑）──────────────────────────────────────────
    # ── 构建 system prompt ────────────────────────────────────────────────
    base_system = llm_config.system_prompt.strip() if llm_config.system_prompt else DEFAULT_SYSTEM_PROMPT
    system_prompt = (base_system + "\n\n" + _CONTEXT_RULES_ADDON) if use_context else base_system

    # ── 构建 messages ─────────────────────────────────────────────────────
    user_content = _build_context_user_content(chunk, context if use_context else None)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # ── LLM 调用 ──────────────────────────────────────────────────────────
    result = _run_chat_json_with_retry(
        llm_config, messages, f"{chunk['title']} 角色分析失败", task_kind="analyze"
    )

    raw_segments = result.get("segments", [])
    if not isinstance(raw_segments, list):
        raise RuntimeError(f"{chunk['title']} 返回的 segments 不是数组")

    known_chars = context.known_characters if context else []
    segments = sanitize_segments(raw_segments, known_characters=known_chars)

    # ── 构建更新后的 context ───────────────────────────────────────────────
    new_speakers = [s["speaker"] for s in segments if s["speaker"] not in ("旁白", "UNKNOWN")]
    extra_new = result.get("new_characters") or []
    updated_known = _merge_known_characters(
        context.known_characters if context else [],
        new_speakers,
        extra_new,
    )

    updated_context = AnalysisContext(
        known_characters=updated_known,
        character_aliases=context.character_aliases if context else {},
        last_context_summary=str(result.get("context_summary") or "").strip(),
        last_active_speakers=[
            str(s).strip() for s in (result.get("active_speakers") or []) if s
        ],
        chunk_index=(context.chunk_index + 1) if context else 1,
    )
    return segments, updated_context


def analyze_chunk_with_retry(chunk: dict[str, str], llm_config: LLMConfig) -> list[dict[str, Any]]:
    """向后兼容接口：无上下文模式分析单个 chunk。"""
    segments, _ = _analyze_chunk_with_context(chunk, llm_config, context=None)
    return segments


def optimize_and_analyze_chunk(chunk: dict[str, str], llm_config: LLMConfig) -> list[dict[str, Any]]:
    system_prompt = llm_config.optimize_analyze_prompt.strip() if llm_config.optimize_analyze_prompt else OPTIMIZE_ANALYZE_PROMPT
    cleaned = basic_tts_text_cleanup(chunk["content"])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"请对下面这段文本一次性完成“语音适配优化 + 角色分析”，并输出 JSON。\n\n标题：{chunk['title']}\n\n正文：\n{cleaned}",
        },
    ]
    result = _run_chat_json_with_retry(llm_config, messages, f"{chunk['title']} 语音适配与角色分析失败", task_kind="analyze")
    segments = result.get("segments", [])
    if not isinstance(segments, list):
        raise RuntimeError(f"{chunk['title']} 返回的 segments 不是数组")
    return sanitize_segments(segments)


def _analyze_chunks_off_mode(
    chunks: list[dict[str, str]], llm_config: LLMConfig
) -> list[dict[str, Any]]:
    """原始并发模式：无跨块上下文，保持向后兼容。"""
    direct_runtime = str(getattr(llm_config, "local_runtime", "") or "").strip().lower() == "direct"
    workers = 1 if direct_runtime else max(1, min(llm_config.workers or 5, len(chunks)))
    results: list[list[dict[str, Any]] | None] = [None] * len(chunks)

    def _work(idx: int, chunk: dict[str, str]) -> tuple[int, list[dict[str, Any]]]:
        return idx, analyze_chunk_with_retry(chunk, llm_config)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_work, i, c) for i, c in enumerate(chunks)]
        for future in as_completed(futures):
            idx, segs = future.result()
            results[idx] = segs

    merged: list[dict[str, Any]] = []
    for item in results:
        if item:
            merged.extend(item)
    return merged


def _analyze_chunks_full_mode(
    chunks: list[dict[str, str]], llm_config: LLMConfig
) -> list[dict[str, Any]]:
    """全文串行模式：上下文跨所有 chunk 传递，质量最高但最慢。"""
    context = AnalysisContext()
    all_segments: list[dict[str, Any]] = []
    for chunk in chunks:
        segs, context = _analyze_chunk_with_context(chunk, llm_config, context)
        all_segments.extend(segs)
    return all_segments


def _analyze_chunks_chapter_mode(
    chunks: list[dict[str, str]], llm_config: LLMConfig
) -> list[dict[str, Any]]:
    """章节内串行 + 章节间并行：推荐默认，兼顾质量与速度。"""
    if not chunks:
        return []

    # 按 chapter_key 对连续 chunk 分组
    groups: list[tuple[str, list[tuple[int, dict[str, str]]]]] = []
    for key, items in groupby(enumerate(chunks), key=lambda t: _chapter_key(t[1]["title"])):
        groups.append((key, list(items)))

    results: list[list[dict[str, Any]] | None] = [None] * len(chunks)

    def _process_group(indexed_chunks: list[tuple[int, dict[str, str]]]) -> None:
        context = AnalysisContext()
        for orig_idx, chunk in indexed_chunks:
            segs, context = _analyze_chunk_with_context(chunk, llm_config, context)
            results[orig_idx] = segs

    direct_runtime = str(getattr(llm_config, "local_runtime", "") or "").strip().lower() == "direct"
    num_groups = len(groups)
    workers = 1 if direct_runtime else max(1, min(llm_config.workers or 5, num_groups))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_group, indexed_chunks) for _, indexed_chunks in groups]
        for future in as_completed(futures):
            future.result()  # 传播异常

    merged: list[dict[str, Any]] = []
    for item in results:
        if item:
            merged.extend(item)
    return merged


def analyze_chunks_with_llm(chunks: list[dict[str, str]], llm_config: LLMConfig) -> list[dict[str, Any]]:
    if not chunks:
        return []

    mode = str(getattr(llm_config, "context_mode", "") or "chapter").strip().lower()
    if mode == "full":
        return _analyze_chunks_full_mode(chunks, llm_config)
    if mode == "off":
        return _analyze_chunks_off_mode(chunks, llm_config)
    # 默认 "chapter"
    return _analyze_chunks_chapter_mode(chunks, llm_config)


def optimize_and_analyze_chunks_with_llm(chunks: list[dict[str, str]], llm_config: LLMConfig) -> list[dict[str, Any]]:
    if not chunks:
        return []

    direct_runtime = str(getattr(llm_config, "local_runtime", "") or "").strip().lower() == "direct"
    workers = 1 if direct_runtime else max(1, min(llm_config.workers or 5, len(chunks)))
    results: list[list[dict[str, Any]] | None] = [None] * len(chunks)

    def _work(index: int, chunk: dict[str, str]) -> tuple[int, list[dict[str, Any]]]:
        return index, optimize_and_analyze_chunk(chunk, llm_config)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_work, idx, chunk) for idx, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            index, segments = future.result()
            results[index] = segments

    merged: list[dict[str, Any]] = []
    for item in results:
        if item:
            merged.extend(item)
    return merged


def resolve_character_aliases(
    segments: list[dict[str, Any]], llm_config: LLMConfig
) -> CharacterRegistry:
    """用 LLM 对 segments 中的角色名做别名归并，返回更新后的 CharacterRegistry。

    如果 llm_config.enable_character_alias_merge=False 则跳过 LLM 调用，仅收集角色名。
    """
    registry = CharacterRegistry()
    registry.observe_segments(segments)
    observed = registry.observed_names()

    # 少于 2 个角色不需要归并
    if not getattr(llm_config, "enable_character_alias_merge", True) or len(observed) < 2:
        return registry

    alias_prompt = (
        llm_config.alias_resolution_prompt.strip()
        if getattr(llm_config, "alias_resolution_prompt", None)
        else CHARACTER_ALIAS_RESOLUTION_PROMPT
    )
    messages = [
        {"role": "system", "content": alias_prompt},
        {"role": "user", "content": f"角色名列表：{', '.join(observed)}"},
    ]
    try:
        result = _run_chat_json_with_retry(
            llm_config, messages, "角色别名归并失败", task_kind="general"
        )
        groups = result.get("groups") or []
        if isinstance(groups, list):
            registry.load_alias_groups(groups)
    except Exception:
        pass  # 归并失败不影响主流程

    return registry


def verify_speakers_pass(
    segments: list[dict[str, Any]],
    llm_config: LLMConfig,
) -> list[dict[str, Any]]:
    """对分析结果做二轮 speaker 核查，采用滑动窗口，只采纳高置信度修正。

    window_size: 每窗口包含的 segment 数
    overlap: 窗口间重叠 segment 数（避免窗口边界漏判）
    """
    if not segments:
        return segments

    verif_prompt = (
        llm_config.speaker_verification_prompt.strip()
        if getattr(llm_config, "speaker_verification_prompt", None)
        else SPEAKER_VERIFICATION_PROMPT
    )
    window_size = 20
    overlap = 5
    step = max(1, window_size - overlap)

    corrections: dict[int, str] = {}  # global_index → suggested_speaker

    i = 0
    while i < len(segments):
        window = segments[i : i + window_size]
        # 只传 speaker + text 给 LLM（节省 token）
        slim = [{"index": j, "speaker": s["speaker"], "text": s["text"]} for j, s in enumerate(window)]
        messages = [
            {"role": "system", "content": verif_prompt},
            {"role": "user", "content": f"请核查以下 segments 的 speaker：\n{slim}"},
        ]
        try:
            result = _run_chat_json_with_retry(
                llm_config, messages, "speaker 核查失败", task_kind="general"
            )
            for corr in (result.get("corrections") or []):
                if str(corr.get("confidence") or "").strip().lower() == "high":
                    local_idx = int(corr.get("index", -1))
                    if 0 <= local_idx < len(window):
                        global_idx = i + local_idx
                        corrections[global_idx] = str(corr.get("suggested_speaker") or "").strip()
        except Exception:
            pass  # 单窗口失败不中断整体
        i += step

    # 应用修正（只接受非空且非 UNKNOWN 的建议）
    for idx, new_speaker in corrections.items():
        if new_speaker and new_speaker != "UNKNOWN" and 0 <= idx < len(segments):
            segments[idx]["speaker"] = new_speaker
            segments[idx].pop("_suspicious", None)  # 修正后清除可疑标记

    return segments


def analyze_text_with_llm(text: str, llm_config: LLMConfig) -> list[dict[str, Any]]:
    """全流程编排：分段 → 优化 → 角色分析 → [别名归并] → [speaker 复核]。"""
    chunks = intelligent_segment_text(text, llm_config)
    optimized_chunks = optimize_chunks_for_tts(chunks, llm_config)
    segments = analyze_chunks_with_llm(optimized_chunks, llm_config)

    # ── 别名归并 ──────────────────────────────────────────────────────────
    if getattr(llm_config, "enable_character_alias_merge", True):
        registry = resolve_character_aliases(segments, llm_config)
        registry.apply_to_segments(segments)

    # ── 二轮 speaker 复核 ─────────────────────────────────────────────────
    if getattr(llm_config, "enable_speaker_verification", False):
        segments = verify_speakers_pass(segments, llm_config)

    return segments
