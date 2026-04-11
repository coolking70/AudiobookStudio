import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from llm_client import OpenAICompatibleClient
from schemas import LLMConfig

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
3. speaker 只填角色名或”旁白”。
4. emotion 只能从以下值中选：neutral, soft, cold, serious, curious, angry
5. style 必须只使用以下英文标签，且用英文逗号+空格分隔：
american accent, australian accent, british accent, canadian accent,
child, chinese accent, elderly, female, high pitch, indian accent,
japanese accent, korean accent, low pitch, male, middle-aged,
moderate pitch, portuguese accent, russian accent, teenager,
very high pitch, very low pitch, whisper, young adult
6. 不要输出任何解释、注释、markdown，只输出 JSON。
7. 如果无法判断，说话内容归为”旁白”。
8. 切分粒度：以”一段旁白叙述”或”一轮对白”为基本单位。相邻旁白叙述可合并为 1-3 句一段，但绝不允许省略或丢弃任何旁白内容。
9. 对白与旁白必须分属不同的 segment。引号内的对白归角色，引号外的叙述（含动作、场景、心理描写）归”旁白”。即使它们在同一个自然句中（如：”你好。”他说。），也必须拆成两个 segment：一个角色对白、一个旁白叙述。
10. 旁白段只包含叙述，不包含引号对白；角色段只包含该角色说的对白，不包含叙述动作。例如 “李明摇了摇头：”不行。””应拆为：旁白”李明摇了摇头：”+ 李明”不行。”。
11. 只有当单段明显过长时，才继续往下细分。
12. 根据角色的性别、年龄、性格特征来选择合适的 style 标签组合。
13. 根据对白的语境和情绪来选择合适的 emotion。
14. **严禁省略任何原文内容**：所有 segments 的 text 按顺序拼接起来必须覆盖原文的每一个字。
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


def sanitize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for seg in segments:
        speaker = str(seg.get("speaker", "旁白")).strip() or "旁白"
        text = str(seg.get("text", "")).strip()
        emotion = str(seg.get("emotion", "neutral")).strip().lower() or "neutral"
        style = normalize_style(seg.get("style"))

        if emotion not in VALID_EMOTIONS:
            emotion = "neutral"
        if not text:
            continue
        if not style:
            style = fallback_style_by_role(speaker, emotion)

        cleaned.append(
            {
                "speaker": speaker,
                "text": text,
                "emotion": emotion,
                "style": style,
                "ref_audio": None,
                "ref_text": None,
            }
        )
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
    return chunks, info


def build_seed_chunks(text: str, llm_config: LLMConfig | None = None, max_chars: int = SEED_CHUNK_MAX_CHARS) -> tuple[list[dict[str, str]], dict[str, Any]]:
    return detect_chapter_structure_with_info(text, llm_config=llm_config, max_chars=max_chars)


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


def analyze_chunk_with_retry(chunk: dict[str, str], llm_config: LLMConfig) -> list[dict[str, Any]]:
    system_prompt = llm_config.system_prompt.strip() if llm_config.system_prompt else DEFAULT_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"请分析下面这段文本，并输出 JSON。\n\n标题：{chunk['title']}\n\n正文：\n{chunk['content']}",
        },
    ]
    result = _run_chat_json_with_retry(llm_config, messages, f"{chunk['title']} 角色分析失败", task_kind="analyze")
    segments = result.get("segments", [])
    if not isinstance(segments, list):
        raise RuntimeError(f"{chunk['title']} 返回的 segments 不是数组")
    return sanitize_segments(segments)


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


def analyze_chunks_with_llm(chunks: list[dict[str, str]], llm_config: LLMConfig) -> list[dict[str, Any]]:
    if not chunks:
        return []

    direct_runtime = str(getattr(llm_config, "local_runtime", "") or "").strip().lower() == "direct"
    workers = 1 if direct_runtime else max(1, min(llm_config.workers or 5, len(chunks)))
    results: list[list[dict[str, Any]] | None] = [None] * len(chunks)

    def _work(index: int, chunk: dict[str, str]) -> tuple[int, list[dict[str, Any]]]:
        return index, analyze_chunk_with_retry(chunk, llm_config)

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


def analyze_text_with_llm(text: str, llm_config: LLMConfig) -> list[dict[str, Any]]:
    chunks = intelligent_segment_text(text, llm_config)
    optimized_chunks = optimize_chunks_for_tts(chunks, llm_config)
    return analyze_chunks_with_llm(optimized_chunks, llm_config)
