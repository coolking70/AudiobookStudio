import json
import html
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from threading import Lock
from typing import Any

from character_registry import CharacterRegistry
from llm_client import LLMContentFilterError, OpenAICompatibleClient
from output_layout import get_temp_archive_dir
from schemas import AnalysisContext, LLMConfig
try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None

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
COMPACT_DECODE_RETRIES = 2
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

DEFAULT_TEXT_PREPROCESS_PROFILE: dict[str, Any] = {
    "pipeline": [
        {"id": "decode_html_entities", "action": "html_unescape", "enabled": True},
        {"id": "base_t2s", "action": "opencc_t2s", "enabled": True},
        {"id": "apply_rules", "action": "apply_ordered_rules", "enabled": True},
    ],
    "preserve": ["祂"],
    "rules": [
        {"id": "phrase_fangfo_01", "match_type": "literal", "source": "彷佛", "target": "仿佛", "priority": 100},
        {"id": "phrase_fangfo_02", "match_type": "literal", "source": "彷彿", "target": "仿佛", "priority": 100},
        {"id": "phrase_fangfo_03", "match_type": "literal", "source": "仿彿", "target": "仿佛", "priority": 100},
        {"id": "char_tiao_after_entity_decode", "match_type": "literal", "source": "䠷", "target": "挑", "priority": 95},
        {"id": "phrase_liaojie", "match_type": "literal", "source": "瞭解", "target": "了解", "priority": 95},
        {"id": "char_jie", "match_type": "literal", "source": "姊", "target": "姐", "priority": 94},
        {"id": "phrase_fanfu", "match_type": "literal", "source": "反覆", "target": "反复", "priority": 90},
        {"id": "char_black", "match_type": "literal", "source": "黒", "target": "黑", "priority": 80},
        {"id": "char_drink", "match_type": "literal", "source": "飮", "target": "饮", "priority": 80},
        {"id": "char_lai", "match_type": "literal", "source": "瀬", "target": "濑", "priority": 80},
        {"id": "char_dan", "match_type": "literal", "source": "弾", "target": "弹", "priority": 80},
        {"id": "char_jian", "match_type": "literal", "source": "剣", "target": "剑", "priority": 80},
        {"id": "char_chuan", "match_type": "literal", "source": "伝", "target": "传", "priority": 80},
        {"id": "char_bu", "match_type": "literal", "source": "歩", "target": "步", "priority": 70},
        {"id": "char_jia", "match_type": "literal", "source": "仮", "target": "假", "priority": 70},
        {"id": "char_quan", "match_type": "literal", "source": "圏", "target": "圈", "priority": 70},
        {"id": "pronoun_ni", "match_type": "literal", "source": "妳", "target": "你", "priority": 60},
        {"id": "pronoun_ta", "match_type": "literal", "source": "牠", "target": "它", "priority": 60},
        {"id": "decorative_separator", "match_type": "regex", "source": r"(?:✽\s*){3,}", "target": "***", "priority": 50},
    ],
}

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
7. 当无法可靠判断说话人时，speaker 填 “UNKNOWN”，不要硬猜成已知角色名；UNKNOWN 会保留给后续人工复核。
8. 切分粒度：以”一段旁白叙述”或”一轮对白”为基本单位。相邻旁白叙述可合并为 1-3 句一段，但绝不允许省略或丢弃任何旁白内容。
9. 对白与旁白必须分属不同的 segment。引号内的对白归角色，引号外的叙述（含动作、表情、场景、心理描写、”说道/问道/答道/苦笑/点头”等提示语）归”旁白”。
10. 遇到 “「对白。」某某说道。” 必须拆成两段：某某 “「对白。」” + 旁白 “某某说道。”；遇到 “某某说道：「对白。」” 必须拆成：旁白 “某某说道：” + 某某 “「对白。」”。
11. 只有当单段明显过长时，才继续往下细分。
12. style 只作为粗略初值；同一 speaker 必须保持稳定性别标签，不要因为情绪把女性角色改成 male，也不要因为严肃语气把男性角色改成 female。
13. 根据对白的语境和情绪来选择合适的 emotion。
14. **严禁省略任何原文内容**：所有 segments 的 text 按顺序拼接起来必须覆盖原文的每一个字。
15. 判断 speaker 时务必结合上下文语义：
    a. “A 向/对 B 发问/说道/回答”之后的对白通常是 A 说的，不是 B；
    b. 内容为”是的/不是/没错/我不知道/嗯”等答复时，说话人优先判断为被提问的 B，而非提问的 A；
    c. 对白中出现”X 大人/殿下/阁下/您”等敬称时，说话人通常不是被称呼的 X 本人；
    d. 不要简单沿用上一轮对白的说话人——每句都要重新根据上下文判断。
    e. 对话密集场景中要维护“当前对话参与者”与“上一句实际说话人”；没有新提示语时，通常在参与者之间轮流，但遇到叙述插入后必须重新判断。
    f. 角色名称是专有名词，speaker 应优先使用原文中出现的完整称呼；不要把“拉菲纳克”这类名字按语义或字面截断成“菲纳克”，除非原文明确只以该称呼出现。
16. 分析每一句前，先判断它的文本功能：
    - 引号内且确实是角色说出口的话：归实际说话人；
    - 非引号行默认归旁白，除非它是明确的内心独白或文本本身就是角色直接话语；
    - “某某说道/问道/笑道/点头/走来/心想/觉得/看着……” 等提示语、动作、心理、表情、场景说明：归旁白；
    - 引号内如果是称号、传闻标题、比喻或被叙述引用的短语（如“像猫狗般水火不容”），不是角色对白，归旁白；
    - “露出『某句话』的表情/神色”这类引号短语不是对白，整句归旁白。
17. speaker 判断要优先找明确提示语和上下文轮次，不要只因为某个角色名离对白最近就归给该角色。无法高置信度判断时填 UNKNOWN。
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
1. 只检查和修正 speaker，必要时可把明显的动作叙述改为旁白；绝对不修改 text、emotion、style 字段。
2. 只在高置信度时才输出修正建议；有疑问则跳过，绝不乱猜。
3. 常见错误场景（优先排查）：
   - “A 向 B 发问”后的对白被标为 B，实应为 A；
   - 回答内容（是的/不是/没错）被标为提问方，实应为被提问方；
   - 对白中称呼”X 大人/殿下”，被错标为 X 本人；
   - 连续轮次中错把上一段的说话人沿用到下一段。
   - 只有动作/心理/提示语的叙述段（如“米拉说道”“堤格尔苦笑着”）被错标成该角色，实应为旁白。
   - 引号内容只是称号、传闻标题、比喻或“露出『某句话』的表情”，被误当成对白。
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

COMPACT_CHAPTER_RULE_DETECTION_PROMPT = """
判断样本文本最像哪种章节标题模式。
可选值仅限：md_chapter_cn / plain_chapter_cn / cn_hui_jie / chapter_en / part_en / numbered / md_heading / none
PATTERN: <pattern> | <confidence> | <一句简短理由>
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
    if speaker in {"旁白", "UNKNOWN"}:
        return "female, moderate pitch"
    gender = infer_speaker_gender(speaker)
    if gender == "female":
        return "female, young adult, moderate pitch"
    if gender == "male":
        return "male, young adult, moderate pitch" if emotion not in {"cold", "serious", "angry"} else "male, low pitch"
    if emotion in {"cold", "serious", "angry"}:
        return "male, low pitch"
    return "male, young adult, moderate pitch"


KNOWN_FEMALE_SPEAKERS = {
    "米拉", "琉德米拉", "琉德米拉·露利叶", "苏菲", "苏菲亚", "艾莲", "艾蕾欧诺拉", "艾蕾欧诺拉·维尔塔利亚",
    "莉姆", "莉姆亚莉夏", "菈娜", "史薇特菈娜", "蒂塔", "凡伦蒂娜", "宓莉莎", "茨魅", "艾榭",
}
KNOWN_MALE_SPEAKERS = {
    "堤格尔", "堤格尔维尔穆德", "堤格尔维尔穆德·冯伦", "拉菲纳克", "加雷宁", "村长", "艾略特",
    "嘉奴隆", "卢里克", "列许", "达马德", "邦纳", "嘉洛诺夫", "塔拉多", "泰纳帝", "罗兰",
    "莱榭克", "萨安", "克雷伊修", "王子", "国王",
}


def infer_speaker_gender(speaker: str) -> str:
    name = str(speaker or "").strip()
    if not name or name in {"旁白", "UNKNOWN"}:
        return "unknown"
    if name in KNOWN_FEMALE_SPEAKERS or any(alias and alias in name for alias in KNOWN_FEMALE_SPEAKERS):
        return "female"
    if name in KNOWN_MALE_SPEAKERS or any(alias and alias in name for alias in KNOWN_MALE_SPEAKERS):
        return "male"
    return "unknown"


def style_with_gender(style: str | None, speaker: str, emotion: str) -> str:
    if str(speaker or "").strip() in {"旁白", "UNKNOWN"}:
        return "female, moderate pitch"
    normalized = normalize_style(style)
    gender = infer_speaker_gender(speaker)
    if not normalized:
        return fallback_style_by_role(speaker, emotion)
    parts = [item.strip() for item in normalized.split(",") if item.strip()]
    if gender in {"female", "male"}:
        parts = [item for item in parts if item not in {"female", "male"}]
        parts.insert(0, gender)
    if not any(item in {"moderate pitch", "low pitch", "high pitch", "very low pitch", "very high pitch"} for item in parts):
        parts.append("moderate pitch")
    dedup: list[str] = []
    for item in parts:
        if item in VALID_INSTRUCT_ITEMS and item not in dedup:
            dedup.append(item)
    return ", ".join(dedup) or fallback_style_by_role(speaker, emotion)


def mark_suspicious_speakers(
    segments: list[dict[str, Any]],
    known_characters: list[str] | None = None,
) -> list[dict[str, Any]]:
    known_set = set(known_characters) if known_characters else set()
    for seg in segments:
        speaker = str(seg.get("speaker") or "旁白").strip() or "旁白"
        seg["speaker"] = speaker
        if speaker == "UNKNOWN":
            seg["_needs_review"] = True
            seg.pop("_suspicious", None)
        elif known_set and speaker != "旁白" and speaker not in known_set:
            seg["_suspicious"] = True
            seg.pop("_needs_review", None)
        else:
            seg.pop("_suspicious", None)
            seg.pop("_needs_review", None)
    return segments


def stabilize_segment_styles(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    speaker_styles: dict[str, dict[str, int]] = {}
    for seg in segments:
        speaker = str(seg.get("speaker") or "旁白")
        style = style_with_gender(seg.get("style"), speaker, str(seg.get("emotion") or "neutral"))
        speaker_styles.setdefault(speaker, {})
        speaker_styles[speaker][style] = speaker_styles[speaker].get(style, 0) + 1

    dominant = {
        speaker: max(counts.items(), key=lambda item: item[1])[0]
        for speaker, counts in speaker_styles.items()
        if counts
    }
    for seg in segments:
        speaker = str(seg.get("speaker") or "旁白")
        emotion = str(seg.get("emotion") or "neutral")
        seg["style"] = style_with_gender(dominant.get(speaker) or seg.get("style"), speaker, emotion)
    return segments


def postprocess_segments(
    segments: list[dict[str, Any]],
    known_characters: list[str] | None = None,
) -> list[dict[str, Any]]:
    """统一收口角色分析结果，只做元数据清洗与音色稳定。"""
    marked = mark_suspicious_speakers(segments, known_characters)
    return stabilize_segment_styles(marked)


def mark_speakers_missing_from_source(
    segments: list[dict[str, Any]],
    source_text: str | None = None,
) -> list[dict[str, Any]]:
    """把原文中完全未出现过的 speaker 标成待复核。"""
    source = str(source_text or "")
    if not source:
        return segments
    for seg in segments:
        speaker = str(seg.get("speaker") or "").strip()
        if not speaker or speaker in {"旁白", "我", "UNKNOWN"}:
            continue
        if speaker not in source:
            seg["_needs_review"] = True
            seg["_suspicious"] = True
            seg["_review_reason"] = "speaker_missing_from_source"
    return segments


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

        speaker = raw_speaker

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
        # 保留所有以 _ 开头的元数据字段（_confidence / _needs_review 等）
        for k, v in seg.items():
            if k.startswith("_"):
                entry[k] = v
        cleaned.append(entry)
    return postprocess_segments(cleaned, list(known_set) if known_set else None)


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


def _split_indexed_chunk_for_compact(
    chunk: dict[str, Any],
    max_chars: int,
    max_lines: int,
) -> list[dict[str, str]]:
    indexed = _ensure_chunk_line_index(chunk)
    lines = indexed.get("lines") or []
    content = str(indexed.get("content", ""))
    if not lines or not content:
        return [chunk]
    source_line_ids = chunk.get("source_line_ids") if isinstance(chunk.get("source_line_ids"), list) else []
    if source_line_ids and len(source_line_ids) == len(lines):
        for line, source_line_id in zip(lines, source_line_ids):
            line["source_line_id"] = str(source_line_id)

    sub_chunks: list[dict[str, str]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        start = int(current[0]["start"])
        end = int(current[-1]["end"])
        piece = content[start:end].strip()
        if piece:
            sub_chunks.append(
                {
                    "title": f"{chunk['title']}（第{len(sub_chunks) + 1}部分）",
                    "content": piece,
                    "source_line_ids": [str(item.get("source_line_id") or item.get("id") or "") for item in current],
                }
            )
        current = []
        current_chars = 0

    for line in lines:
        line_text = str(line.get("text", ""))
        line_len = max(1, len(line_text))
        need_flush = bool(
            current and (
                len(current) >= max_lines or
                current_chars + line_len > max_chars
            )
        )
        if need_flush:
            flush()
        current.append(line)
        current_chars += line_len

    flush()
    return sub_chunks or [chunk]


def _is_timeout_like_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return any(token in message for token in ("超时", "timeout", "timed out"))


def _is_content_filter_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return isinstance(exc, LLMContentFilterError) or "content_filter" in message or "内容过滤" in message


def _is_length_cutoff_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return "finish_reason=length" in message or "输出被截断" in message


def _make_llm_skipped_segments(
    chunk: dict[str, Any],
    reason: str,
    detail: str,
) -> list[dict[str, Any]]:
    """Preserve filtered text as reviewable UNKNOWN segments instead of aborting the job."""
    indexed = _ensure_chunk_line_index(chunk)
    lines = indexed.get("lines") or []
    title = str(indexed.get("title") or "未命名片段")
    source_line_ids = chunk.get("source_line_ids") if isinstance(chunk.get("source_line_ids"), list) else []
    segments: list[dict[str, Any]] = []
    if not lines:
        text = str(indexed.get("content") or "").strip()
        if not text:
            return []
        return [{
            "speaker": "UNKNOWN",
            "text": text,
            "emotion": "neutral",
            "style": "female, moderate pitch",
            "_confidence": "low",
            "_needs_review": True,
            "_llm_skipped": True,
            "_skip_reason": reason,
            "_source_title": title,
            "_source_lines": "",
            "_evidence": f"{title} 触发 {reason}，已保留原文并跳过 LLM 角色归属。{detail}",
        }]

    for idx, line in enumerate(lines):
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        line_id = str(source_line_ids[idx]) if idx < len(source_line_ids) else str(line.get("id") or "")
        segments.append({
            "speaker": "UNKNOWN",
            "text": text,
            "emotion": "neutral",
            "style": "female, moderate pitch",
            "_confidence": "low",
            "_needs_review": True,
            "_llm_skipped": True,
            "_skip_reason": reason,
            "_source_title": title,
            "_source_lines": line_id,
            "_evidence": f"{title} {line_id} 触发 {reason}，已保留原文并跳过 LLM 角色归属。{detail}",
        })
    return segments


class AnalyzeDiagnostics:
    """Collects compact-analysis fallback events for frontend feedback."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.events: list[dict[str, Any]] = []
        self.stats: Counter[str] = Counter()

    def record(
        self,
        event_type: str,
        title: str,
        message: str,
        *,
        split_into: int | None = None,
        source_line_ids: list[str] | None = None,
        detail: str | None = None,
    ) -> None:
        event = {
            "type": event_type,
            "title": title,
            "message": message,
        }
        if split_into is not None:
            event["split_into"] = int(split_into)
        if source_line_ids:
            event["source_line_ids"] = [str(item) for item in source_line_ids if str(item or "").strip()]
        if detail:
            event["detail"] = detail
        with self._lock:
            self.stats[event_type] += 1
            if len(self.events) < 80:
                self.events.append(event)

    def to_dict(self, segments: list[dict[str, Any]]) -> dict[str, Any]:
        needs_review = sum(1 for seg in (segments or []) if seg.get("_needs_review"))
        unknown = sum(1 for seg in (segments or []) if str(seg.get("speaker") or "").strip().upper() == "UNKNOWN")
        with self._lock:
            return {
                "stats": dict(self.stats),
                "events": list(self.events),
                "needs_review_count": needs_review,
                "unknown_count": unknown,
            }


def _is_transient_llm_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    transient_tokens = (
        "超时",
        "timeout",
        "timed out",
        "连接失败",
        "connection",
        "主动断开",
        "remote protocol",
        "server disconnected",
        "返回为空",
        "temporarily",
        "try again",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    return any(token in message for token in transient_tokens)


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


@lru_cache(maxsize=1)
def _load_text_preprocess_profile() -> dict[str, Any]:
    profile_path = Path(__file__).resolve().parent / "docs" / "zh_hans_rules_runtime.json"
    if profile_path.exists():
        try:
            loaded = json.loads(profile_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception as exc:
            logging.warning("[preprocess] 读取规则文件失败，已回退内置规则：%s", exc)
    return DEFAULT_TEXT_PREPROCESS_PROFILE


@lru_cache(maxsize=1)
def _get_opencc_converter() -> Any | None:
    if OpenCC is None:
        return None
    try:
        return OpenCC("t2s")
    except Exception as exc:
        logging.warning("[preprocess] OpenCC 初始化失败，已跳过简繁统一：%s", exc)
        return None


@lru_cache(maxsize=1)
def _get_compiled_preprocess_rules() -> list[dict[str, Any]]:
    profile = _load_text_preprocess_profile()
    raw_rules = profile.get("rules") if isinstance(profile, dict) else []
    compiled: list[dict[str, Any]] = []
    for rule in sorted(raw_rules or [], key=lambda item: int(item.get("priority", 0)), reverse=True):
        source = str(rule.get("source") or "")
        if not source:
            continue
        entry = {
            "id": str(rule.get("id") or source),
            "match_type": str(rule.get("match_type") or "literal").strip().lower(),
            "source": source,
            "target": str(rule.get("target") or ""),
            "priority": int(rule.get("priority", 0)),
        }
        if entry["match_type"] == "regex":
            try:
                entry["pattern"] = re.compile(source)
            except re.error as exc:
                logging.warning("[preprocess] 跳过无效正则规则 %s：%s", entry["id"], exc)
                continue
        compiled.append(entry)
    return compiled


def _protect_preserve_tokens(text: str, tokens: list[str]) -> tuple[str, dict[str, str]]:
    protected = str(text or "")
    restore_map: dict[str, str] = {}
    for idx, token in enumerate(tokens):
        if not token or token not in protected:
            continue
        placeholder = f"\uE000PRESERVE_{idx}\uE001"
        protected = protected.replace(token, placeholder)
        restore_map[placeholder] = token
    return protected, restore_map


def _restore_preserve_tokens(text: str, restore_map: dict[str, str]) -> str:
    restored = str(text or "")
    for placeholder, original in restore_map.items():
        restored = restored.replace(placeholder, original)
    return restored


def preprocess_text_for_chapter_detection(text: str) -> tuple[str, dict[str, Any]]:
    """章节识别预处理：实体解码、简繁统一、补充规则替换。"""
    original = str(text or "")
    profile = _load_text_preprocess_profile()
    preserve_tokens = [str(item) for item in profile.get("preserve", []) if str(item or "")]
    working, restore_map = _protect_preserve_tokens(original, preserve_tokens)

    info: dict[str, Any] = {
        "applied": False,
        "original_chars": len(original),
        "processed_chars": len(original),
        "html_unescape_changed": False,
        "opencc_changed": False,
        "rule_hits": [],
    }

    pipeline = profile.get("pipeline") if isinstance(profile, dict) else []
    enabled_actions = {
        str(step.get("action") or "").strip().lower()
        for step in (pipeline or [])
        if isinstance(step, dict) and step.get("enabled", True)
    }

    if "html_unescape" in enabled_actions:
        updated = html.unescape(working)
        info["html_unescape_changed"] = updated != working
        working = updated

    if "opencc_t2s" in enabled_actions:
        converter = _get_opencc_converter()
        if converter is not None:
            updated = converter.convert(working)
            info["opencc_changed"] = updated != working
            working = updated

    if "apply_ordered_rules" in enabled_actions:
        for rule in _get_compiled_preprocess_rules():
            before = working
            if rule["match_type"] == "regex":
                working, count = rule["pattern"].subn(rule["target"], working)
            else:
                count = before.count(rule["source"])
                if count:
                    working = before.replace(rule["source"], rule["target"])
            if count:
                info["rule_hits"].append({"id": rule["id"], "count": count})

    working = _restore_preserve_tokens(working, restore_map)
    normalized = working.replace("\r\n", "\n").strip()
    info["processed_chars"] = len(normalized)
    info["applied"] = normalized != original.strip()
    return normalized, info


def infer_heading_pattern_with_llm(sample_text: str, llm_config: LLMConfig) -> dict[str, str]:
    if _is_compact_output_mode(llm_config):
        messages = [
            {"role": "system", "content": COMPACT_CHAPTER_RULE_DETECTION_PROMPT},
            {
                "role": "user",
                "content": f"请判断下面文本样本最像哪一种章节标题模式。\n\n文本样本：\n{sample_text}",
            },
        ]
        content = _run_chat_text_with_retry(llm_config, messages, purpose="章节规则探测")
        _dump_compact_debug("compact_chapter_detect", "chapter_detect", messages, response_text=content)
        line = next((row.strip() for row in content.splitlines() if row.strip().startswith("PATTERN:")), "")
        if not line:
            raise RuntimeError(f"章节规则探测返回格式无效：{content[:200]}")
        payload = line[len("PATTERN:"):].strip()
        parts = [part.strip() for part in payload.split("|", 2)]
        pattern = parts[0] if parts else "none"
        confidence = parts[1] if len(parts) > 1 else ""
        reason = parts[2] if len(parts) > 2 else ""
        return {
            "pattern": pattern or "none",
            "confidence": confidence,
            "reason": reason,
        }

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
    normalized, preprocess_info = preprocess_text_for_chapter_detection(text)
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
        "preprocess": preprocess_info,
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


def _run_chat_text_with_retry(
    llm_config: LLMConfig,
    messages: list[dict[str, Any]],
    purpose: str,
    max_tokens: int | None = None,
) -> str:
    """对 compact/text 调用增加瞬时错误重试。

    不重试 finish_reason=length 这类确定性容量错误，避免重复浪费 token。
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        client = OpenAICompatibleClient(llm_config)
        try:
            return client.chat_text(messages, max_tokens=max_tokens, purpose=purpose)
        except Exception as exc:
            last_error = exc
            if "finish_reason=length" in str(exc):
                break
            if attempt >= MAX_RETRIES or not _is_transient_llm_error(exc):
                break
            logging.warning(
                "[llm] %s 第 %s/%s 次调用失败（%s），稍后重试",
                purpose,
                attempt,
                MAX_RETRIES,
                exc,
            )
            time.sleep(RETRY_DELAY_SECONDS * attempt)
    assert last_error is not None
    raise last_error


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
    if _is_compact_output_mode(llm_config):
        compact_chunk = _ensure_chunk_line_index(seed_chunk)
        numbered_text = str(compact_chunk.get("numbered_text") or "")
        line_count = len(compact_chunk.get("lines") or [])
        # LM Studio / OpenAI 兼容本地服务在超长 numbered_text 上容易直接 400。
        # 这种情况下先用本地保守切块，再让后续更小的块继续尝试 compact。
        if len(numbered_text) >= 6000 or line_count >= 120:
            fallback_chunks = split_chunk_if_too_long(seed_chunk, max_chars=char_budget)
            if len(fallback_chunks) > 1:
                logging.info(
                    f"[compact] {seed_chunk['title']} 输入过长（chars={len(numbered_text)}, lines={line_count}），先本地切成 {len(fallback_chunks)} 块再继续。"
                )
                return fallback_chunks
        prompt = (
            f"{COMPACT_SEGMENT_PLAN_PROMPT}\n\n"
            f"补充要求：单个 chunk 的 content 尽量控制在 {char_budget} 字以内；如果自然分段后某块略超出也可以接受，但不要过度细碎。"
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"请对下面文本做智能分段规划。\n\n标题：{seed_chunk['title']}\n\n{compact_chunk['numbered_text']}",
            },
        ]
        try:
            compact_text = _run_chat_text_with_retry(llm_config, messages, purpose="紧凑智能分段")
            _dump_compact_debug("compact_segment", seed_chunk["title"], messages, response_text=compact_text)
            chunks = _decode_compact_chunk_plan(compact_text, compact_chunk, seed_chunk["title"])
            if chunks:
                return chunks
        except Exception as exc:
            fallback_chunks = split_chunk_if_too_long(seed_chunk, max_chars=char_budget)
            if len(fallback_chunks) > 1:
                logging.warning(
                    f"[compact] {seed_chunk['title']} 智能分段失败（{exc}），已改用本地保守切块 {len(fallback_chunks)} 段。"
                )
                _dump_compact_debug("compact_segment_error", seed_chunk["title"], messages, response_text=compact_text if 'compact_text' in locals() else "", error=str(exc))
                return fallback_chunks
            logging.warning(
                f"[compact] {seed_chunk['title']} 智能分段失败（{exc}），保留本地切块结果继续 compact 流程。"
            )
            _dump_compact_debug("compact_segment_error", seed_chunk["title"], messages, response_text=compact_text if 'compact_text' in locals() else "", error=str(exc))
            return fallback_chunks or [seed_chunk]

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
    if _is_compact_output_mode(llm_config):
        planned = intelligent_segment_seed_chunk(seed_chunk, llm_config)
        return [optimize_chunk_for_tts(chunk, llm_config) for chunk in planned]

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
    if _is_compact_output_mode(llm_config):
        compact_chunk = _ensure_chunk_line_index({"title": chunk["title"], "content": cleaned})
        messages = [
            {"role": "system", "content": COMPACT_OPTIMIZATION_PROMPT},
            {
                "role": "user",
                "content": f"请优化下面这段文本，使其更适合 OmniVoice 合成。\n\n标题：{chunk['title']}\n\n{compact_chunk['numbered_text']}",
            },
        ]
        try:
            compact_text = _run_chat_text_with_retry(llm_config, messages, purpose="紧凑文本优化")
            _dump_compact_debug("compact_optimize", chunk["title"], messages, response_text=compact_text)
            optimized_content = _decode_compact_optimized_content(compact_text, compact_chunk)
            if optimized_content:
                return {"title": chunk["title"], "content": optimized_content}
        except Exception as exc:
            logging.warning(
                f"[compact] {chunk['title']} 语音适配优化失败（{exc}），保留基础清理文本继续 compact 流程。"
            )
            _dump_compact_debug("compact_optimize_error", chunk["title"], messages, response_text=compact_text if 'compact_text' in locals() else "", error=str(exc))
            return {"title": chunk["title"], "content": cleaned}

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
输入是带行号小说文本。判断每行 speaker 之前必须先检查以下三大陷阱：

═══ 陷阱A：被谈论 ≠ 说话人 ═══
对白中出现以下任一情况时，说话人**绝对不是**该名字/被称呼者：
  · 出现称谓「少爷/大人/殿下/阁下/陛下/老师/师父/姐姐/哥哥」→ 说话人不是被称呼方
  · 出现第三方名字+助词「的事/的话/怎样/如何/为人」→ 说话人不是该名字本人
  · 用第三人称提及某角色（”她/他+名字”）→ 说话人不是该角色
示例：
  「还是要请少爷多考虑一下蒂塔的事。」
  → 正确：拉菲纳克（”少爷”=堤格尔被称呼→不是堤格尔；”蒂塔的事”=蒂塔被谈论→不是蒂塔）
  → 错误：蒂塔（× 仅看名字出现就归给该名字）

═══ 陷阱B：紧跟的叙述行揭示真实说话人 ═══
对白下一行若出现”某某挥手/点头/否定/笑道/沉吟/抱头/转身”等动作或情绪描写，**这个主语就是上一行对白的说话人**——不要被前文谁在提问/前文焦点干扰。
示例：
  L1: 「隔壁邻居，难道是莱德梅里兹吗？」
  L2: [中间插入旁白介绍背景]
  L3: 「不，不是那边的邻居。」
  L4: 村长大大地挥手，否定了堤格尔的猜测。
  → 正确：L3 = 村长（L4 明确”村长否定堤格尔的猜测”，L3 就是被否定的回答）
  → 错误：L3 = 堤格尔（× 仅因前文 L1 是堤格尔在问就让他自答）

═══ 陷阱C：动作叙述行（无引号）必为 0=旁白 ═══
形如”XXX喝了一口酒，把话题转回来：”、”XXX抓了抓头发，焦躁地呼出一口气”这类句子，**没有引号 → 必是旁白**。即使主语是某角色、即使后面跟了冒号引出对白也不例外，冒号是旁白的过渡标点，不是该角色在说话。
示例：
  L1: 堤格尔喝了一口陶杯中的酒，把话题转回来：
  L2: 「说到战姬大人……」
  → 正确：L1 = 0（旁白，描写动作）；L2 = 堤格尔
  → 错误：L1 = 堤格尔（× 即便提到他，没引号就不算他说话）

──────────────────────────────────────────────

完成上述检查后再按以下规则输出：

ROLES: <编号>=<角色>|<style>, ...
SEGS:
<行号或范围> <编号> <情绪> <置信度> [<锚点>|rest]

规则：
- 0=旁白，?=UNKNOWN，其他编号必须先出现在 ROLES
- 情绪只可用 ne so co se cu an
- 置信度只可用 h（高，有明确提示语/动作主语指向，三大陷阱已排除）m（中，从对话轮次推断）l（低，仍有歧义或难以排除陷阱）；? 必须配 l
- style 只能从以下短标签选 1-3 个并用空格分隔：male female child teenager young_adult middle-aged elderly low_pitch moderate_pitch high_pitch very_low_pitch very_high_pitch whisper
- style 禁止写性格、身份、职业、剧情描述或长形容词
- ROLES 示例：`ROLES: 1=堤格尔|male young_adult moderate_pitch, 2=拉菲纳克|male young_adult moderate_pitch`
- 所有输入行必须按顺序完整覆盖；相邻旁白可合并，如 L1-L3 0 ne h
- 一行里同时含对白和提示语时必须拆段：`「对白」某某说道` 写 `Lx 角色 ne h “「对白」”` 再写 `Lx 0 ne h rest`
- 动作、表情、心理、提示语（说道/问道/答道/苦笑/点头/走来/心想/觉得等）默认 0=旁白
- 非引号行默认 0=旁白
- 引号内不一定都是对白：称号、传闻、比喻、”露出「某句话」的表情”都归 0=旁白
- 没有新提示语的连续对白，通常在当前参与者之间轮流；但插入旁白后要根据提示语重新判断
- 角色名优先用原文完整称呼；不确定用 ?
- 锚点必须与原文完全一致
- 不输出 JSON、markdown 或解释
""".strip()

COMPACT_SEGMENT_PLAN_PROMPT = """
输入是带行号文本。按顺序规划 chunk。
CHUNKS:
<行号范围> | <标题>
...
要求：完整覆盖，不重叠；优先章节/场景/自然段/完整对白；标题尽量短。
""".strip()

COMPACT_OPTIMIZATION_PROMPT = f"""
输入是带行号文本。只做必要的 OmniVoice 保守优化。
OPT:
<行号范围> =
<单行号> ~ <优化后的文本>
...
规则：每行都要覆盖；无改动用 `=`；改写只能单行；不得删句。
可用标签：{", ".join(SUPPORTED_NON_VERBAL_TAGS)}
拼音可用声调数字。
""".strip()

COMPACT_ALIAS_RESOLUTION_PROMPT = """
输入是一组角色名。只在非常确定同一人时归并。
GROUPS:
<规范名> | <别名1>, <别名2>
...
没有可归并项时只输出 `GROUPS:`
""".strip()

COMPACT_SPEAKER_VERIFICATION_PROMPT = """
核查 speaker 是否明显有误，只在高置信度时修正。
FIXES:
<窗口内index> | <建议speaker>
...
优先修正：纯动作/心理/提示语被标为角色时改为旁白；非对白引号被误当对白时改为旁白；对白 speaker 明显错位时改为实际说话人。
无修正时只输出 `FIXES:`；不要修改 text/emotion/style。
""".strip()

# 情绪代码 → 全称映射
_EMOTION_CODES: dict[str, str] = {
    "ne": "neutral", "so": "soft", "co": "cold",
    "se": "serious", "cu": "curious", "an": "angry",
    # 容错：允许全称
    "neutral": "neutral", "soft": "soft", "cold": "cold",
    "serious": "serious", "curious": "curious", "angry": "angry",
}

# 置信度代码 → 全称映射
_CONFIDENCE_CODES: dict[str, str] = {
    "h": "high", "m": "medium", "l": "low",
    # 容错：允许全称
    "high": "high", "medium": "medium", "low": "low",
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
        # 格式（新）：line_ref  speaker  emotion  confidence  [anchor_or_rest]
        # 格式（旧）：line_ref  speaker  emotion  [anchor_or_rest]
        tokens = raw_line.split(None, 4)  # 最多 5 个 token
        if len(tokens) < 3:
            continue

        line_ref_token, speaker_token, emotion_token = tokens[0], tokens[1], tokens[2]

        # 判断第 4 个 token 是置信度代码（h/m/l）还是旧格式的锚点/rest
        raw_token3 = tokens[3].strip() if len(tokens) > 3 else ""
        if raw_token3 in _CONFIDENCE_CODES:
            confidence = _CONFIDENCE_CODES[raw_token3]
            anchor_token = tokens[4].strip() if len(tokens) > 4 else ""
        else:
            # 旧格式（无置信度字段），默认 high
            confidence = "high"
            anchor_token = raw_token3

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
                "_confidence": confidence,
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


def _is_compact_output_mode(llm_config: LLMConfig) -> bool:
    return str(getattr(llm_config, "output_mode", "verbose") or "verbose").lower() == "compact"


def _ensure_chunk_line_index(chunk: dict[str, Any]) -> dict[str, Any]:
    if chunk.get("lines") and chunk.get("numbered_text"):
        return chunk
    result = number_lines_with_soft_split(chunk.get("content", ""))
    return {**chunk, "numbered_text": result["numbered_text"], "lines": result["lines"]}


def _parse_line_ref_token(line_ref_token: str, lines_index: list[dict[str, Any]]) -> list[str]:
    raw_token = (line_ref_token or "").strip()
    normalized_token = raw_token.strip("<>[](){}").replace(" ", "")
    if re.match(r"^\d+(?:-\d+)?$", normalized_token):
        if "-" in normalized_token:
            start_num, end_num = normalized_token.split("-", 1)
            normalized_token = f"L{start_num}-L{end_num}"
        else:
            normalized_token = f"L{normalized_token}"
    elif re.match(r"^L\d+-\d+$", normalized_token, re.I):
        start_part, end_num = normalized_token.split("-", 1)
        normalized_token = f"{start_part.upper()}-L{end_num}"
    else:
        normalized_token = normalized_token.upper()

    match = re.match(r"^(L\d+)(?:-(L\d+))?$", normalized_token)
    if not match:
        raise ValueError(f"无法解析行号范围：{line_ref_token}")
    start_id = match.group(1)
    end_id = match.group(2) or start_id
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
        raise ValueError(f"行号范围没有命中任何输入行：{line_ref_token}")
    return ids_in_range


def _decode_compact_chunk_plan(text: str, seed_chunk: dict[str, Any], fallback_title: str) -> list[dict[str, str]]:
    chunk = _ensure_chunk_line_index(seed_chunk)
    lines_index: list[dict[str, Any]] = chunk.get("lines", [])
    if not lines_index:
        raise ValueError("chunk 缺少 lines 索引，无法解码 compact 分段结果")

    content = str(chunk.get("content", ""))
    line_by_id = {ln["id"]: ln for ln in lines_index}
    ordered_ids = [ln["id"] for ln in lines_index]
    parsed_chunks: list[dict[str, str]] = []
    covered_ids: list[str] = []

    for raw_line in text.splitlines():
        row = raw_line.strip()
        if not row or row.startswith("CHUNKS:"):
            continue
        parts = [part.strip() for part in row.split("|", 1)]
        if not parts or not parts[0]:
            continue
        ids_in_range = _parse_line_ref_token(parts[0], lines_index)
        start_ln = line_by_id[ids_in_range[0]]
        end_ln = line_by_id[ids_in_range[-1]]
        chunk_text = content[start_ln["start"]:end_ln["end"]].strip()
        if not chunk_text:
            continue
        title = parts[1] if len(parts) > 1 and parts[1] else f"{fallback_title} - 第{len(parsed_chunks) + 1}块"
        parsed_chunks.append({"title": title, "content": chunk_text})
        covered_ids.extend(ids_in_range)

    if not parsed_chunks:
        raise ValueError("compact 分段结果为空，可能格式有误")
    if covered_ids != ordered_ids:
        raise ValueError("compact 分段结果未能按顺序完整覆盖所有输入行")
    return parsed_chunks


def _decode_compact_optimized_content(text: str, raw_chunk: dict[str, Any]) -> str:
    chunk = _ensure_chunk_line_index(raw_chunk)
    lines_index: list[dict[str, Any]] = chunk.get("lines", [])
    if not lines_index:
        raise ValueError("chunk 缺少 lines 索引，无法解码 compact 优化结果")

    replacements: dict[str, str] = {}
    covered_ids: list[str] = []
    ordered_ids = [ln["id"] for ln in lines_index]
    line_by_id = {ln["id"]: ln for ln in lines_index}

    for raw_line in text.splitlines():
        row = raw_line.strip()
        if not row or row.startswith("OPT:"):
            continue
        if " ~ " in row:
            line_ref_token, replacement = row.split(" ~ ", 1)
            ids_in_range = _parse_line_ref_token(line_ref_token.strip(), lines_index)
            if len(ids_in_range) != 1:
                raise ValueError("compact 优化改写只允许作用于单行")
            lid = ids_in_range[0]
            replacements[lid] = replacement.strip()
            covered_ids.append(lid)
            continue
        if row.endswith("="):
            line_ref_token = row[:-1].strip()
            ids_in_range = _parse_line_ref_token(line_ref_token, lines_index)
            for lid in ids_in_range:
                replacements[lid] = line_by_id[lid]["text"]
            covered_ids.extend(ids_in_range)
            continue
        raise ValueError(f"无法解析 compact 优化行：{row}")

    if covered_ids != ordered_ids:
        raise ValueError("compact 优化结果未能按顺序完整覆盖所有输入行")

    original_content = str(chunk.get("content", ""))
    rebuilt_parts: list[str] = []
    cursor = 0
    for ln in lines_index:
        start = int(ln["start"])
        end = int(ln["end"])
        rebuilt_parts.append(original_content[cursor:start])
        rebuilt_parts.append(replacements.get(ln["id"], ln["text"]))
        cursor = end
    rebuilt_parts.append(original_content[cursor:])
    return "".join(rebuilt_parts).strip()


def _decode_compact_alias_groups(text: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        row = raw_line.strip()
        if not row or row.startswith("GROUPS:"):
            continue
        parts = [part.strip() for part in row.split("|", 1)]
        canonical = parts[0] if parts else ""
        aliases = []
        if len(parts) > 1 and parts[1]:
            aliases = [item.strip() for item in parts[1].split(",") if item.strip()]
        if canonical and aliases:
            groups.append({"canonical": canonical, "aliases": aliases})
    return groups


def _decode_compact_speaker_fixes(text: str) -> list[dict[str, Any]]:
    fixes: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        row = raw_line.strip()
        if not row or row.startswith("FIXES:"):
            continue
        parts = [part.strip() for part in row.split("|", 1)]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        speaker = parts[1]
        if speaker:
            fixes.append({"index": idx, "suggested_speaker": speaker})
    return fixes


def _dump_compact_debug(stage: str, title: str, messages: list[dict[str, Any]], response_text: str = "", error: str = "") -> None:
    try:
        out_dir = get_temp_archive_dir("llm_debug")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_title = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", (title or stage))[:80]
        target = out_dir / f"{stamp}_{stage}_{safe_title}.json"
        target.write_text(json.dumps({
            "stage": stage,
            "title": title,
            "messages": messages,
            "response_text": response_text,
            "error": error,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _build_compact_user_content(chunk: dict[str, Any], context: AnalysisContext | None) -> str:
    """构建 compact user 内容。发送 numbered_text，而非原始 content。"""
    parts: list[str] = []
    if context is not None and (
        context.known_characters or context.last_context_summary or context.last_active_speakers
    ):
        ctx_lines = ["CTX"]
        if context.known_characters:
            ctx_lines.append(f"角色: {', '.join(context.known_characters)}")
        if context.last_context_summary:
            ctx_lines.append(f"摘要: {context.last_context_summary}")
        if context.last_active_speakers:
            ctx_lines.append(f"活跃: {', '.join(context.last_active_speakers)}")
        parts.append("\n".join(ctx_lines))

    numbered = chunk.get("numbered_text") or chunk.get("content", "")
    parts.append(f"标题: {chunk['title']}\n{numbered}")
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
    diagnostics: AnalyzeDiagnostics | None = None,
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

        compact_char_budget = max(
            900,
            min(1600, int(((llm_config.max_tokens or 0) * 0.8) or 1400)),
        )
        compact_line_budget = 48
        if (
            len(chunk.get("lines") or []) > compact_line_budget
            or len(str(chunk.get("numbered_text") or "")) > compact_char_budget * 2
        ):
            sub_chunks = _split_indexed_chunk_for_compact(
                chunk,
                max_chars=compact_char_budget,
                max_lines=compact_line_budget,
            )
            if len(sub_chunks) > 1:
                logging.info(
                    "[compact] %s 输入较大（%s 行），先切成 %s 个更小子块继续角色分析",
                    chunk["title"],
                    len(chunk.get("lines") or []),
                    len(sub_chunks),
                )
                merged_segments: list[dict[str, Any]] = []
                current_context = context
                for sub_chunk in sub_chunks:
                    sub_segments, current_context = _analyze_chunk_with_context(
                        sub_chunk, llm_config, current_context, diagnostics
                    )
                    merged_segments.extend(sub_segments)
                return merged_segments, current_context or AnalysisContext()

        system_prompt = COMPACT_ANALYSIS_PROMPT

        user_content = _build_compact_user_content(chunk, context if use_context else None)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        compact_text = ""
        try:
            last_decode_error: Exception | None = None
            for decode_attempt in range(1, COMPACT_DECODE_RETRIES + 1):
                compact_text = _run_chat_text_with_retry(llm_config, messages, purpose="紧凑角色分析")
                _dump_compact_debug("compact_analyze", chunk["title"], messages, response_text=compact_text)
                try:
                    segments = _decode_compact_output(compact_text, chunk)
                    break
                except Exception as decode_exc:
                    last_decode_error = decode_exc
                    if decode_attempt >= COMPACT_DECODE_RETRIES:
                        raise
                    logging.warning(
                        "[compact] %s 第 %s/%s 次解码失败（%s），原块重试一次",
                        chunk["title"],
                        decode_attempt,
                        COMPACT_DECODE_RETRIES,
                        decode_exc,
                    )
                    time.sleep(RETRY_DELAY_SECONDS * decode_attempt)
        except Exception as exc:
            if _is_content_filter_error(exc):
                _dump_compact_debug(
                    "compact_analyze_content_filter",
                    chunk["title"],
                    messages,
                    response_text=compact_text,
                    error=str(exc),
                )
                sub_chunks = _split_indexed_chunk_for_compact(
                    chunk,
                    max_chars=320,
                    max_lines=1,
                )
                if len(sub_chunks) > 1:
                    if diagnostics is not None:
                        diagnostics.record(
                            "content_filter_split",
                            chunk["title"],
                            "触发服务端内容过滤，已切成行级子块继续分析",
                            split_into=len(sub_chunks),
                            source_line_ids=[str(item) for item in (chunk.get("source_line_ids") or []) if str(item or "").strip()],
                            detail=str(exc),
                        )
                    logging.warning(
                        "[compact] %s 触发服务端内容过滤，已切成 %s 个行级子块继续分析，过滤行将标记为待补充",
                        chunk["title"],
                        len(sub_chunks),
                    )
                    merged_segments: list[dict[str, Any]] = []
                    current_context = context
                    for sub_chunk in sub_chunks:
                        sub_segments, current_context = _analyze_chunk_with_context(
                            sub_chunk, llm_config, current_context, diagnostics
                        )
                        merged_segments.extend(sub_segments)
                    return merged_segments, current_context or AnalysisContext()

                if diagnostics is not None:
                    diagnostics.record(
                        "content_filter_skip",
                        chunk["title"],
                        "触发服务端内容过滤且无法继续细分，已标记为待补充",
                        source_line_ids=[str(item) for item in (chunk.get("source_line_ids") or []) if str(item or "").strip()],
                        detail=str(exc),
                    )
                logging.warning(
                    "[compact] %s 触发服务端内容过滤且无法继续细分，已标记为 UNKNOWN 待补充",
                    chunk["title"],
                )
                skipped = sanitize_segments(
                    _make_llm_skipped_segments(
                        chunk,
                        "content_filter",
                        str(exc),
                    ),
                    known_characters=context.known_characters if context else [],
                )
                return skipped, context or AnalysisContext()

            if _is_timeout_like_error(exc):
                sub_chunks = _split_indexed_chunk_for_compact(
                    chunk,
                    max_chars=max(800, compact_char_budget - 200),
                    max_lines=max(24, compact_line_budget - 12),
                )
                if len(sub_chunks) > 1:
                    if diagnostics is not None:
                        diagnostics.record(
                            "timeout_split",
                            chunk["title"],
                            "请求超时，已切成更小子块继续 compact 角色分析",
                            split_into=len(sub_chunks),
                            source_line_ids=[str(item) for item in (chunk.get("source_line_ids") or []) if str(item or "").strip()],
                            detail=str(exc),
                        )
                    logging.warning(
                        "[compact] %s 超时（%s），已切成 %s 个更小子块继续 compact 角色分析",
                        chunk["title"],
                        exc,
                        len(sub_chunks),
                    )
                    _dump_compact_debug(
                        "compact_analyze_error",
                        chunk["title"],
                        messages,
                        response_text=compact_text,
                        error=str(exc),
                    )
                    merged_segments: list[dict[str, Any]] = []
                    current_context = context
                    for sub_chunk in sub_chunks:
                        sub_segments, current_context = _analyze_chunk_with_context(
                            sub_chunk, llm_config, current_context, diagnostics
                        )
                        merged_segments.extend(sub_segments)
                    return merged_segments, current_context or AnalysisContext()
            _dump_compact_debug("compact_analyze_error", chunk["title"], messages, response_text=compact_text, error=str(exc))
            if _is_length_cutoff_error(exc):
                split_max_chars = 320
                split_max_lines = 6
                skip_reason = "finish_reason_length"
                split_message = "输出 token 不足"
            else:
                split_max_chars = max(500, compact_char_budget // 2)
                split_max_lines = max(8, compact_line_budget // 2)
                skip_reason = "compact_decode_failed"
                split_message = "解码失败"
            sub_chunks = _split_indexed_chunk_for_compact(
                chunk,
                max_chars=split_max_chars,
                max_lines=split_max_lines,
            )
            if len(sub_chunks) > 1:
                if diagnostics is not None:
                    diagnostics.record(
                        "length_split" if _is_length_cutoff_error(exc) else "decode_split",
                        chunk["title"],
                        f"{split_message}，已切成更小子块继续 compact 角色分析",
                        split_into=len(sub_chunks),
                        source_line_ids=[str(item) for item in (chunk.get("source_line_ids") or []) if str(item or "").strip()],
                        detail=str(exc),
                    )
                logging.warning(
                    "[compact] %s %s（%s），已切成 %s 个更小子块继续 compact 角色分析",
                    chunk["title"],
                    split_message,
                    exc,
                    len(sub_chunks),
                )
                merged_segments: list[dict[str, Any]] = []
                current_context = context
                for sub_chunk in sub_chunks:
                    sub_segments, current_context = _analyze_chunk_with_context(
                        sub_chunk, llm_config, current_context, diagnostics
                    )
                    merged_segments.extend(sub_segments)
                return merged_segments, current_context or AnalysisContext()
            if _is_length_cutoff_error(exc):
                if diagnostics is not None:
                    diagnostics.record(
                        "length_skip",
                        chunk["title"],
                        "输出 token 不足且无法继续细分，已标记为待补充",
                        source_line_ids=[str(item) for item in (chunk.get("source_line_ids") or []) if str(item or "").strip()],
                        detail=str(exc),
                    )
                logging.warning(
                    "[compact] %s 输出 token 不足且无法继续细分，已标记为 UNKNOWN 待补充",
                    chunk["title"],
                )
                skipped = sanitize_segments(
                    _make_llm_skipped_segments(
                        chunk,
                        skip_reason,
                        str(exc),
                    ),
                    known_characters=context.known_characters if context else [],
                )
                return skipped, context or AnalysisContext()
            raise RuntimeError(
                f"{chunk['title']} 紧凑角色分析失败：模型输出不符合 compact 格式，"
                f"且当前块已无法继续细分。原始错误：{exc}"
            ) from exc

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
    if _is_compact_output_mode(llm_config):
        optimized = optimize_chunk_for_tts(chunk, llm_config)
        return analyze_chunk_with_retry(optimized, llm_config)

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
    chunks: list[dict[str, str]], llm_config: LLMConfig, diagnostics: AnalyzeDiagnostics | None = None
) -> list[dict[str, Any]]:
    """原始并发模式：无跨块上下文，保持向后兼容。"""
    direct_runtime = str(getattr(llm_config, "local_runtime", "") or "").strip().lower() == "direct"
    workers = 1 if direct_runtime else max(1, min(llm_config.workers or 5, len(chunks)))
    results: list[list[dict[str, Any]] | None] = [None] * len(chunks)

    def _work(idx: int, chunk: dict[str, str]) -> tuple[int, list[dict[str, Any]]]:
        segs, _ = _analyze_chunk_with_context(chunk, llm_config, context=None, diagnostics=diagnostics)
        return idx, segs

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
    chunks: list[dict[str, str]], llm_config: LLMConfig, diagnostics: AnalyzeDiagnostics | None = None
) -> list[dict[str, Any]]:
    """全文串行模式：上下文跨所有 chunk 传递，质量最高但最慢。"""
    context = AnalysisContext()
    all_segments: list[dict[str, Any]] = []
    for chunk in chunks:
        segs, context = _analyze_chunk_with_context(chunk, llm_config, context, diagnostics)
        all_segments.extend(segs)
    return all_segments


def _analyze_chunks_chapter_mode(
    chunks: list[dict[str, str]], llm_config: LLMConfig, diagnostics: AnalyzeDiagnostics | None = None
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
            segs, context = _analyze_chunk_with_context(chunk, llm_config, context, diagnostics)
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


def analyze_chunks_with_llm(
    chunks: list[dict[str, str]],
    llm_config: LLMConfig,
    diagnostics: AnalyzeDiagnostics | None = None,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    mode = str(getattr(llm_config, "context_mode", "") or "chapter").strip().lower()
    if mode == "full":
        return postprocess_segments(_analyze_chunks_full_mode(chunks, llm_config, diagnostics))
    if mode == "off":
        return postprocess_segments(_analyze_chunks_off_mode(chunks, llm_config, diagnostics))
    # 默认 "chapter"
    return postprocess_segments(_analyze_chunks_chapter_mode(chunks, llm_config, diagnostics))


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
    return postprocess_segments(merged)


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
        if _is_compact_output_mode(llm_config):
            compact_messages = [
                {"role": "system", "content": COMPACT_ALIAS_RESOLUTION_PROMPT},
                {"role": "user", "content": f"角色名列表：{', '.join(observed)}"},
            ]
            compact_text = _run_chat_text_with_retry(llm_config, compact_messages, purpose="紧凑别名归并")
            _dump_compact_debug("compact_alias", "alias_resolution", compact_messages, response_text=compact_text)
            registry.load_alias_groups(_decode_compact_alias_groups(compact_text))
        else:
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
        try:
            if _is_compact_output_mode(llm_config):
                compact_messages = [
                    {"role": "system", "content": COMPACT_SPEAKER_VERIFICATION_PROMPT},
                    {"role": "user", "content": f"请核查以下 segments 的 speaker：\n{slim}"},
                ]
                compact_text = _run_chat_text_with_retry(llm_config, compact_messages, purpose="紧凑说话人复核")
                _dump_compact_debug("compact_verify", f"speaker_verify_{i}", compact_messages, response_text=compact_text)
                compact_fixes = _decode_compact_speaker_fixes(compact_text)
                for corr in compact_fixes:
                    local_idx = int(corr.get("index", -1))
                    if 0 <= local_idx < len(window):
                        global_idx = i + local_idx
                        corrections[global_idx] = str(corr.get("suggested_speaker") or "").strip()
            else:
                messages = [
                    {"role": "system", "content": verif_prompt},
                    {"role": "user", "content": f"请核查以下 segments 的 speaker：\n{slim}"},
                ]
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

    return postprocess_segments(segments)


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

    final_segments = postprocess_segments(segments)
    return mark_speakers_missing_from_source(final_segments, source_text=text)
