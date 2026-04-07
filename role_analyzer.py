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
你是一个“小说/对白文本分析器”。
请把用户提供的原文分析成适合 TTS 的 JSON 结构。

要求：
1. 输出必须是合法 JSON 对象。
2. 格式固定为：
{
  "segments": [
    {
      "speaker": "旁白",
      "text": "夜色降临。",
      "emotion": "neutral",
      "style": "female, moderate pitch"
    }
  ]
}
3. speaker 只填角色名或“旁白”。
4. emotion 只能从以下值中选：neutral, soft, cold, serious, curious, angry
5. style 必须只使用以下英文标签，且用英文逗号+空格分隔：
american accent, australian accent, british accent, canadian accent,
child, chinese accent, elderly, female, high pitch, indian accent,
japanese accent, korean accent, low pitch, male, middle-aged,
moderate pitch, portuguese accent, russian accent, teenager,
very high pitch, very low pitch, whisper, young adult
6. 不要输出任何解释、注释、markdown，只输出 JSON。
7. 如果无法判断，说话内容归为“旁白”。
8. 优先按“自然段 / 完整对白回合”切分，不要切得过碎。
9. 旁白连续叙述可合并成 1-3 句，只要语义完整即可。
10. 只有当单段明显过长时，才继续往下细分。
11. 根据角色的性别、年龄、性格特征来选择合适的 style 标签组合。
12. 根据对白的语境和情绪来选择合适的 emotion。
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
2. content 必须直接来自用户原文，只允许做极少量空白整理，不要改写内容，不要补写，不要删减事实。
3. 优先按章节、场景切换、自然段和完整对白回合切分。
4. 每个 chunk 尽量语义完整，不要把一句对白拆开。
5. 如果用户原文里有章节标题，优先保留到对应 chunk 的 title。
6. 如果当前文本本来就很短，可以只返回 1 个 chunk。
7. 不要输出解释、注释或 markdown。
""".strip()

TEXT_OPTIMIZATION_PROMPT = f"""
你是一个“OmniVoice 文本语音适配优化器”。
你的任务是在不改变原意、剧情、角色归属的前提下，把输入文本整理成更适合 OmniVoice 合成的版本。

请参考 OmniVoice 官方能力：
1. 支持内联非语言标签：{", ".join(SUPPORTED_NON_VERBAL_TAGS)}
2. 中文发音纠正可用带声调数字的拼音，如 ZHE2、SHE2、ZHE1。

优化原则：
1. 尽量保持原句顺序、段落结构和含义不变。
2. 识别明显的语气词、叹息、笑声、疑问尾音，可在必要时替换成 OmniVoice 支持的标签。
3. 识别多音字、容易误读的人名地名、生僻字，可在必要时改写为 OmniVoice 可读的拼音形式（使用带声调数字的拼音）。
4. 将 OmniVoice 可能误解的特殊符号做保守替换，例如把【】替换成更自然的中文表达或标点。
5. 在容易导致连读的空白换行之间补入必要标点，但不要过度加标点。
6. 不要随意改写文学风格；如果没有必要优化，就尽量少改。
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


def split_by_chapters(text: str) -> list[dict[str, str]]:
    pattern = r"^(#{1,2}\s*第\s*\d+\s*章[：:]?\s*.*)$"
    parts = re.split(pattern, text, flags=re.MULTILINE)

    chunks = []
    index = 1
    while index < len(parts) - 1:
        title = parts[index].strip().lstrip("#").strip()
        content = parts[index + 1].strip()
        if content:
            chunks.append({"title": title, "content": content})
        index += 2

    if not chunks:
        normalized = text.strip()
        if normalized:
            chunks.append({"title": "全文", "content": normalized})
    return chunks


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


def build_seed_chunks(text: str, max_chars: int = SEED_CHUNK_MAX_CHARS) -> list[dict[str, str]]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    chapters = split_by_chapters(normalized)
    chunks = []
    for chapter in chapters:
        chunks.extend(split_chunk_if_too_long(chapter, max_chars=max_chars))
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
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        client = OpenAICompatibleClient(llm_config)
        try:
            return client.chat_json(messages)
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(RETRY_DELAY_SECONDS * attempt)
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
    char_budget = max(1800, min(DEFAULT_CHUNK_MAX_CHARS, int(llm_config.max_tokens * 2.4)))
    prompt = (
        f"{SEGMENT_PLAN_PROMPT}\n\n"
        f"补充要求：单个 chunk 的 content 尽量控制在 {char_budget} 字以内；如果自然分段后某块略超出也可以接受，但不要过度细碎。"
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"请对下面文本做智能分段规划。\n\n标题：{seed_chunk['title']}\n\n正文：\n{seed_chunk['content']}",
        },
    ]
    result = _run_chat_json_with_retry(llm_config, messages, f"{seed_chunk['title']} 智能分段失败")
    chunks = _coerce_chunks(result.get("chunks", []), seed_chunk["title"])
    if chunks:
        return chunks
    return split_chunk_if_too_long(seed_chunk, max_chars=char_budget)


def intelligent_segment_text(text: str, llm_config: LLMConfig) -> list[dict[str, str]]:
    seed_chunks = build_seed_chunks(text)
    if not seed_chunks:
        return []

    workers = max(1, min(llm_config.workers or 5, len(seed_chunks)))
    results: list[list[dict[str, str]] | None] = [None] * len(seed_chunks)

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
    return merged


def optimize_chunk_for_tts(chunk: dict[str, str], llm_config: LLMConfig) -> dict[str, str]:
    cleaned = basic_tts_text_cleanup(chunk["content"])
    messages = [
        {"role": "system", "content": TEXT_OPTIMIZATION_PROMPT},
        {
            "role": "user",
            "content": f"请优化下面这段文本，使其更适合 OmniVoice 合成。\n\n标题：{chunk['title']}\n\n正文：\n{cleaned}",
        },
    ]
    result = _run_chat_json_with_retry(llm_config, messages, f"{chunk['title']} 语音适配优化失败")
    optimized_chunks = _coerce_chunks(result.get("chunks", []), chunk["title"])
    if optimized_chunks:
        return optimized_chunks[0]
    return {"title": chunk["title"], "content": cleaned}


def optimize_chunks_for_tts(chunks: list[dict[str, str]], llm_config: LLMConfig) -> list[dict[str, str]]:
    if not chunks:
        return []

    workers = max(1, min(llm_config.workers or 5, len(chunks)))
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
    result = _run_chat_json_with_retry(llm_config, messages, f"{chunk['title']} 角色分析失败")
    segments = result.get("segments", [])
    if not isinstance(segments, list):
        raise RuntimeError(f"{chunk['title']} 返回的 segments 不是数组")
    return sanitize_segments(segments)


def analyze_chunks_with_llm(chunks: list[dict[str, str]], llm_config: LLMConfig) -> list[dict[str, Any]]:
    if not chunks:
        return []

    workers = max(1, min(llm_config.workers or 5, len(chunks)))
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


def analyze_text_with_llm(text: str, llm_config: LLMConfig) -> list[dict[str, Any]]:
    chunks = intelligent_segment_text(text, llm_config)
    optimized_chunks = optimize_chunks_for_tts(chunks, llm_config)
    return analyze_chunks_with_llm(optimized_chunks, llm_config)
