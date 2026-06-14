"""Unit test for script/group-chat block splitting (2026-06-14).

群组聊天/剧本格式节（如『五女神的房间』）以「说话人：台词」逐行书写、无「」引号，
旧流程把整块吞成单个旁白段。本测试锁定 split 行为 + 别名归一 + 不误伤普通散文/引用式。

Run: python tools/test_script_block_split.py   (no API needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "BookVoiceParser"))

from book_voice_parser.alias_registry import AliasRegistry  # noqa: E402
from book_voice_parser.schema import AttributionType, SegmentEx  # noqa: E402
from book_voice_parser.script_block import (  # noqa: E402
    apply_script_block_split,
    is_script_block,
    split_script_block_text,
)

BLOCK = (
    "（4） 其之一 女王：总之，我们终于宣战了。 姬百合：不愧是小卑弥！ "
    "女王：这下我们已经无法回头了……呵呵，呵呵呵…… 女王：我快吐了。 "
    "miki：mi～kimi～ki～！ miki：miki！ mi～ki～！ 女王：美姬同学怎么了？ "
    "姬百合：不太清楚，但她最近好像很喜欢这样…… 女王：好吵啊！ 姬百合：咦？"
)
PROSE = "所谓的人际关系，就好比不断选择正确答案的游戏一样。比方说朋友开了个玩笑。"
QUOTED = "纱月同学转过头说道：「你也差不多该让我听听你的答覆了吧。」我吓了一跳。"

ROSTER = {"高田卑弥呼": ["女王", "小卑弥", "卑弥呼"],
          "羽贺铃兰": ["姬百合"], "根本美姬": ["miki", "美姬同学"]}
REG = AliasRegistry.from_role_hints(ROSTER)


def test_detection() -> None:
    assert is_script_block(BLOCK) is True
    assert is_script_block(PROSE) is False, "普通散文不应判为脚本块"
    assert is_script_block(QUOTED) is False, "引用式「某某：「台词」」不应判为脚本块"


def test_split_and_alias() -> None:
    pieces = split_script_block_text(BLOCK)
    assert pieces[0] == (None, "（4） 其之一"), pieces[0]
    labels = [n for n, _ in pieces if n]
    assert labels[:3] == ["女王", "姬百合", "女王"]
    assert REG.canonicalize("女王") == "高田卑弥呼"
    assert REG.canonicalize("姬百合") == "羽贺铃兰"
    assert REG.canonicalize("miki") == "根本美姬"


def test_apply_splits_narration_only() -> None:
    segs = [
        SegmentEx(speaker="甘织玲奈子", text="五女神的房间", quote_id="q0205",
                  attribution_type=AttributionType.IMPLICIT, confidence=0.6),
        SegmentEx(speaker="旁白", text=BLOCK, quote_id="q0206_n1",
                  attribution_type=AttributionType.NARRATOR, confidence=1.0),
    ]
    out, stats = apply_script_block_split(segs, REG.canonicalize)
    assert stats["blocks"] == 1 and stats["lines"] >= 10
    speakers = [s.speaker for s in out]
    assert "高田卑弥呼" in speakers and "羽贺铃兰" in speakers and "根本美姬" in speakers
    # 头部「五女神的房间」段未被动
    assert out[0].speaker == "甘织玲奈子" and out[0].text == "五女神的房间"
    # 前言「（4） 其之一」→ 旁白
    assert any(s.speaker == "旁白" and "其之一" in s.text for s in out)


def test_no_false_split() -> None:
    # 普通旁白段 / 引用式 / 单个冒号都不应被切
    for txt in (PROSE, QUOTED, "时间：下午三点，我赶到了车站。"):
        segs = [SegmentEx(speaker="旁白", text=txt, quote_id="qx",
                          attribution_type=AttributionType.NARRATOR, confidence=1.0)]
        out, stats = apply_script_block_split(segs, REG.canonicalize)
        assert len(out) == 1 and stats["blocks"] == 0, f"不应切分: {txt[:20]}"


def main() -> None:
    tests = [test_detection, test_split_and_alias,
             test_apply_splits_narration_only, test_no_false_split]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
