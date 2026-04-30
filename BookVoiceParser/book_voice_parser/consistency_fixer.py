"""
consistency_fixer.py
--------------------
对 parse_novel() 输出的 SegmentEx 列表做全局一致性修正。

修正逻辑按保守程度从强到弱：
  1. 场景约束     —— 说话人不在 scene_characters 时降置信度
  2. 连续双人轮替 —— AB 对话中同一人连续说 ≥3 句，且对方在场，则翻转
  3. 跳变检测     —— 说话人突然换成与前后完全无关的角色时降置信度
  4. 旁白/未知下压 —— 原有逻辑保留，低置信旁白进一步压分

设计原则：
  - 不覆写 confidence ≥ HIGH_CONF 的片段（规则层已高度确信的显式归属）
  - 修正后保留原 evidence，追加修正说明
  - 所有修正只降分或翻转，不会凭空提高 confidence 到高置信区间
"""

from __future__ import annotations

import re

from .schema import AttributionType, SegmentEx


# ── 阈值常量 ──────────────────────────────────────────────────────────────
HIGH_CONF = 0.90       # 高置信归属，不参与一致性修正
SCENE_PENALTY = 0.15   # 说话人不在场景时扣分
JUMP_PENALTY = 0.12    # 突变跳变时扣分
MIN_CONF = 0.10        # 置信度下限

SKIP_SPEAKERS = {"旁白", "未知", "众人", "大家", "二人", "三人", "所有人"}


# ── 辅助函数 ──────────────────────────────────────────────────────────────
def _is_fixable(seg: SegmentEx) -> bool:
    """只对低于高置信阈值、且说话人有意义的片段做修正。"""
    return seg.confidence < HIGH_CONF and seg.speaker not in SKIP_SPEAKERS


def _clamp(value: float) -> float:
    return max(MIN_CONF, min(1.0, value))


def _append_evidence(seg: SegmentEx, note: str) -> None:
    seg.evidence = f"{seg.evidence or ''}；{note}" if seg.evidence else note


# ── 修正 1：场景约束 ──────────────────────────────────────────────────────
def _apply_scene_constraint(segments: list[SegmentEx], narrator: str | None = None) -> None:
    """
    若 scene_characters 非空且说话人不在其中，降低 confidence。
    scene_characters 由 candidate_gen 根据上下文人名窗口生成，
    不在场景里的人说话大概率是误判。
    叙述者（narrator）以第一人称叙事，名字不会出现在自己的上下文窗口，豁免此约束。
    """
    for seg in segments:
        if not _is_fixable(seg):
            continue
        if not seg.scene_characters:
            continue
        if seg.speaker in seg.scene_characters:
            continue
        if narrator and seg.speaker == narrator:
            continue  # 叙述者豁免：其名字不出现在上下文是正常现象，不扣分
        # 说话人不在场景中，扣分
        seg.confidence = _clamp(seg.confidence - SCENE_PENALTY)
        _append_evidence(seg, f"场景约束：{seg.speaker} 不在 scene_characters")


# ── 修正 2：连续双人对话轮替 ──────────────────────────────────────────────
def _apply_alternation(segments: list[SegmentEx]) -> None:
    """
    检测 AAAB 模式（A 连续说 ≥3 句后，B 说一句，然后 A 又连续说多句）。
    当满足以下条件时，把中间 A 的连续片段翻转为 B：
      - 两人都在 scene_characters 中
      - 翻转目标的 confidence < HIGH_CONF
      - 翻转后 speaker 在 scene_characters 中

    这是最激进的修正，额外要求置信度 < 0.65 才操作。
    """
    n = len(segments)
    i = 0
    while i < n:
        seg = segments[i]
        if seg.confidence >= 0.65 or seg.speaker in SKIP_SPEAKERS:
            i += 1
            continue

        # 找这个说话人的连续片段长度
        j = i
        while j < n and segments[j].speaker == seg.speaker and segments[j].confidence < HIGH_CONF:
            j += 1
        run_len = j - i
        if run_len < 3:
            i += 1
            continue

        # 找前面最近的不同说话人
        prev_speaker: str | None = None
        for k in range(i - 1, -1, -1):
            sp = segments[k].speaker
            if sp not in SKIP_SPEAKERS and sp != seg.speaker:
                prev_speaker = sp
                break

        # 找后面最近的不同说话人
        next_speaker: str | None = None
        for k in range(j, n):
            sp = segments[k].speaker
            if sp not in SKIP_SPEAKERS and sp != seg.speaker:
                next_speaker = sp
                break

        # 只有前后都是同一个人，且该人在场景中，才翻转中间片段
        if prev_speaker and prev_speaker == next_speaker:
            # 检查场景约束
            scene = seg.scene_characters
            if not scene or (prev_speaker in scene and seg.speaker in scene):
                # 翻转连续段中最靠近中间的那些（保留首尾各一句）
                flip_start = i + 1
                flip_end = j - 1
                for fi in range(flip_start, flip_end):
                    s = segments[fi]
                    if s.confidence < 0.65:
                        s.speaker = prev_speaker
                        s.confidence = _clamp(s.confidence - 0.05)  # 仍保守降分
                        s.attribution_type = AttributionType.IMPLICIT
                        _append_evidence(s, f"连续轮替修正 → {prev_speaker}")
        i = j


# ── 修正 3：跳变检测 ──────────────────────────────────────────────────────
def _apply_jump_detection(segments: list[SegmentEx]) -> None:
    """
    若某片段的说话人与前 3 句和后 3 句里的说话人集合完全不重合，
    认为可能是跳变，降低置信度。
    """
    n = len(segments)
    for i, seg in enumerate(segments):
        if not _is_fixable(seg) or seg.confidence < 0.5:
            continue

        # 收集前后窗口说话人
        window: set[str] = set()
        for k in range(max(0, i - 3), i):
            sp = segments[k].speaker
            if sp not in SKIP_SPEAKERS:
                window.add(sp)
        for k in range(i + 1, min(n, i + 4)):
            sp = segments[k].speaker
            if sp not in SKIP_SPEAKERS:
                window.add(sp)

        # 如果窗口非空且当前说话人不在窗口中，视为跳变
        if window and seg.speaker not in window:
            seg.confidence = _clamp(seg.confidence - JUMP_PENALTY)
            _append_evidence(seg, f"跳变：{seg.speaker} 与前后窗口说话人 {sorted(window)} 无交集")


# ── 修正 4：旁白/未知置信度下压（原有逻辑保留）────────────────────────────
def _apply_unknown_penalty(segments: list[SegmentEx]) -> None:
    """低置信度旁白/未知片段进一步压分，确保它们优先进入复核队列。"""
    for seg in segments:
        if seg.speaker not in {"未知", "旁白"}:
            continue
        if seg.confidence > 0.5:
            continue
        seg.confidence = _clamp(min(seg.confidence, 0.35))


# ── 修正 5：双人场景重复归因惩罚 ─────────────────────────────────────────
def _apply_two_person_repeat_penalty(segments: list[SegmentEx]) -> None:
    """
    针对日式双人对话的特定失败模式：

    当场景只有两个角色（A, B）且存在连续两条以上的低置信度台词都归给 A 时，
    将第二条起的重复者置信度下调，并附加 needs_review 标记。
    不做翻转（翻转过于激进），只是降分让它们进入人工复核队列。

    条件：
    - scene_characters 恰好包含 2 个非旁白角色
    - 连续 2+ 条相同说话人
    - 均无 explicit 类型的归属（implicit / unknown）
    - confidence < HIGH_CONF
    """
    IMPLICIT_TYPES = {AttributionType.IMPLICIT, AttributionType.UNKNOWN}
    REPEAT_PENALTY = 0.10

    n = len(segments)
    for i in range(1, n):
        prev = segments[i - 1]
        curr = segments[i]

        if curr.confidence >= HIGH_CONF:
            continue
        if curr.speaker in SKIP_SPEAKERS or prev.speaker in SKIP_SPEAKERS:
            continue
        if curr.speaker != prev.speaker:
            continue
        if curr.attribution_type not in IMPLICIT_TYPES:
            continue

        # 判断是否双人场景
        scene = curr.scene_characters or []
        real_scene = [s for s in scene if s not in SKIP_SPEAKERS]
        if len(real_scene) != 2:
            continue

        other = next((s for s in real_scene if s != curr.speaker), None)
        if other is None:
            continue

        # 同说话人连续重复（无显式归属）→ 降分，提示可能应为对方
        curr.confidence = _clamp(curr.confidence - REPEAT_PENALTY)
        _append_evidence(
            curr,
            f"双人重复惩罚：{curr.speaker} 连续发言，可能应为 {other}，建议复核"
        )


# ── 修正 6：受话人陷阱检测（规则辅助）──────────────────────────────────────
_ADDRESSEE_RE = re.compile(
    r"(?:对着?|朝|冲|向|望着?|看向|注视着?)\s*"
    r"(?P<name>[一-龥]{2,6})"
    r"[一-龥，,\s]{0,20}?"
    r"(?:说道|说|道|问道|问|答道|喊道)"
    r"[：:，,「\s]*$"
)


def _apply_addressee_trap_check(segments: list[SegmentEx]) -> None:
    """
    若前文含「对着/朝 + X + 说道」且当前说话人 == X，说明 X 是受话人而非说话人。
    降低置信度，附加提示。

    注意：只处理 confidence < HIGH_CONF 的片段，避免误伤高置信显式归属。
    """
    for seg in segments:
        if not _is_fixable(seg):
            continue
        cb = seg.context_before or ""
        m = _ADDRESSEE_RE.search(cb[-100:])
        if not m:
            continue
        mentioned = m.group("name")
        # 如果当前说话人的名字以 mentioned 结尾（简称匹配），说明是受话人陷阱
        if seg.speaker == mentioned or seg.speaker.endswith(mentioned) or seg.speaker.startswith(mentioned):
            seg.confidence = _clamp(seg.confidence - 0.15)
            _append_evidence(
                seg,
                f"受话人陷阱：前文「{m.group().strip()[:20]}」中 {mentioned} 是受话对象"
            )


# ── 主入口 ────────────────────────────────────────────────────────────────
def fix_consistency(segments: list[SegmentEx], narrator: str | None = None) -> list[SegmentEx]:
    """
    对 SegmentEx 列表做全局一致性修正（in-place）。

    顺序：场景约束 → 受话人陷阱 → 跳变检测 → 双人重复惩罚 → 连续轮替 → 旁白下压
    """
    _apply_scene_constraint(segments, narrator=narrator)
    _apply_addressee_trap_check(segments)
    _apply_jump_detection(segments)
    _apply_two_person_repeat_penalty(segments)
    _apply_alternation(segments)
    _apply_unknown_penalty(segments)
    return segments
