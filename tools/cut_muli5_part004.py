# -*- coding: utf-8 -*-
"""从 第五卷/原文.txt 切出 part_004 对应段（确定性，可复现）。

part_004 = 快照 segments 2251-3000（slice(2250,3000)）。
原文切割区间 raw[105374:139198]，紧接 part_003 末尾（part_003 止于 seg2250
「给我一个吻」），止于 seg3000 文本「我，真的，不太想去……」。

校验：切出的文本须含 seg2251 首句 与 seg3000 末句，字数 ~33.8k。
用法：python tools/cut_muli5_part004.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VOL5 = REPO / "docs" / "samples" / "第五卷"
SNAP = Path(r"I:\code\aitts\text\task_snapshot_segments_2026-06-12_optimized.json")

START, END = 105374, 139198  # part_003 末尾 → seg3000 末


def main() -> None:
    raw = (VOL5 / "原文.txt").read_text(encoding="utf-8")
    seg = json.loads(SNAP.read_text(encoding="utf-8"))["segments"]
    chunk = raw[START:END].strip()

    first = (seg[2250].get("text") or "").strip()  # seg2251
    last = (seg[2999].get("text") or "").strip()   # seg3000
    assert first and first in chunk, f"首句(seg2251)未命中: {first!r}"
    assert last and last in chunk, f"末句(seg3000)未命中: {last!r}"

    out = VOL5 / "原文_part004.txt"
    out.write_text(chunk, encoding="utf-8")
    print(f"写出 {out.name}：{len(chunk)} 字")
    print(f"  首(seg2251): {chunk[:42]!r}")
    print(f"  末(seg3000): ...{chunk[-42:]!r}")


if __name__ == "__main__":
    main()
