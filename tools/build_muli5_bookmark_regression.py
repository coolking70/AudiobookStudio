#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 muli4 书签回归样本的口径，切出 muli_5_part_001 的人工书签纠错集。

与 muli4 的差异：本卷只有一份「生成时实际使用的」优化快照（raw 即 reviewed），
用户是对照成品音频逐处下书签的，因此 raw_speaker == reviewed_speaker，
expected_speaker 为人工判定的正确说话人（与之不同即为一处归因错误）。
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LRC = ROOT / "outputs" / "muli_5_parts" / "muli_5_part_001.lrc"
SNAP = Path(r"I:\code\aitts\text\task_snapshot_segments_2026-06-12_optimized.json")
BM = Path(r"E:\Temp\audiobook_bookmarks_export(1).json")
SRC_FULL = ROOT / "docs" / "samples" / "第五卷" / "原文.txt"
SRC_PART = ROOT / "docs" / "samples" / "第五卷" / "原文_part001.txt"
OUT = ROOT / "docs" / "samples" / "第五卷" / "muli5_part001_bookmark_regression.json"

RAW_FIELDS = ["speaker", "text", "quote_id", "confidence", "evidence",
              "attribution_type", "candidates", "scene_characters"]


def hhmmss(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def load_lrc():
    pat = re.compile(r"\[(\d+):(\d+)\.(\d+)\](.*)")
    out = []
    for ln in LRC.read_text(encoding="utf-8").splitlines():
        m = pat.match(ln)
        if m:
            mm, ss, cs = int(m[1]), int(m[2]), int(m[3])
            out.append(((mm * 60 + ss) + cs / 100.0, m[4]))
    return out


def lrc_entry(L, i):
    t, txt = L[i]
    return {"start": round(t, 2), "time": hhmmss(t), "text": txt}


def idx_at(L, sec):
    cur = 0
    for i, (t, _) in enumerate(L):
        if t <= sec:
            cur = i
        else:
            break
    return cur


# (1-based LRC 行号): (正确说话人, 判据)
CORR = {
    187: ("甘织玲奈子", "下句旁白「我抬起脸…小心翼翼地询问」，「咦……？」是玲奈子的反应"),
    281: ("高田卑弥呼", "conf0.2「LLM未返回」；真唯在下文283才回应「原来如此」，此句为挑战方宣战"),
    310: ("照泽耀子", "自我介绍「我是照泽耀子」，且称对方「玲奈子同学」，非玲奈子本人"),
    311: ("甘织玲奈子", "下文旁白「我情不自禁地喊了出来」，是玲奈子的惊呼"),
    314: ("照泽耀子", "装帅的耀子续话，夸「玲奈子同学很可爱」，非高田卑弥呼"),
    318: ("照泽耀子", "「和玲奈子同学一起加油」，称呼对方=非玲奈子本人"),
    319: ("甘织玲奈子", "被强势凑近，玲奈子求饶「请、请手下留情」"),
    324: ("旁白", "引号外叙述（attribution_type=narrator、conf1）却挂到真唯"),
    325: ("照泽耀子", "「玲奈子同学，再见啰！」=对方道别"),
    326: ("甘织玲奈子", "「好、好的，再见」+下句旁白「我轻轻地挥手回应」=玲奈子"),
    330: ("旁白", "栏目/群组名「五女神的房间」"),
    334: ("旁白", "玲奈子独白中假设的举例台词"),
    336: ("旁白", "叙述「这就是所谓的『看气氛』」"),
    361: ("甘织玲奈子", "「我遵照你的要求，打电话给你了」=打 Morning Call 的玲奈子"),
    362: ("濑名紫阳花", "「心跳好快」是紫阳花调侃，下句363玲奈子接「真的很快」"),
    367: ("甘织玲奈子", "称「紫阳花同学，祝你…」=玲奈子的挂断语"),
    389: ("濑名紫阳花", "第三人称「小玲奈躺在旁边」，是紫阳花追忆旅行"),
    398: ("甘织玲奈子", "397/399 皆玲奈子独白，「不妙」属同一段被切断"),
    400: ("濑名紫阳花", "「现在不是姐姐吧？」带嘟嘴，下句401玲奈子听到而心跳加速"),
    428: ("甘织玲奈子", "遥奈尚未登场；玲奈子调侃紫阳花温泉时玩姐姐游戏被真唯看到"),
    429: ("濑名紫阳花", "紫阳花冷静附和，下句旁白「为什么她能那么冷静地附和我的话」"),
    431: ("甘织玲奈子", "同上，遥奈未登场，玲奈子续话"),
    455: ("甘织玲奈子", "上句旁白「我清楚地用言语表达」，主语是玲奈子；遥奈未登场"),
    465: ("甘织玲奈子", "「抱歉，吵到你了吗」=玲奈子，下句466遥奈回「姐姐…更吵」"),
    473: ("甘织遥奈", "遥奈问「你刚才在和谁讲电话」，下句474玲奈子答「和紫阳花同学」"),
    486: ("甘织玲奈子", "玲奈子在心里重复遥奈刚说的「顺理成章」，下句「我懂她的意思」"),
    490: ("甘织遥奈", "玲奈子想象妹妹（遥奈）去问紫阳花的台词"),
    506: ("旁白", "卡牌名「姐姐的强权」，玲奈子翻牌朗读"),
    511: ("甘织遥奈", "「你不愿意说我就不问了」=放弃追问的遥奈，下句512玲奈子「这才对嘛」"),
    514: ("甘织遥奈", "遥奈怀疑「你好像在隐瞒什么不好的事」，下句515玲奈子反驳"),
    530: ("甘织玲奈子", "玲奈子主张妹妹论，下句531旁白「我极力主张」=玲奈子"),
    556: ("甘织玲奈子", "纱月问玲奈子接吻事，玲奈子反问「为什么纱月同学要问那种事情」"),
    573: ("甘织玲奈子", "玲奈子的担忧回应「应该会吧」"),
    590: ("琴纱月", "此场景仅玲奈子+纱月，真唯为第三人称，此句为纱月吐槽"),
    604: ("琴纱月", "「我之后也会叮咛真唯」真唯为第三人称（系统已标 suspicious）"),
    651: ("琴纱月", "「只是为了与真唯一决胜负才玩」=纱月（evidence 亦写『符合纱月口吻』却挂错）"),
    661: ("甘织玲奈子", "「纱月同学之前玩过的FPS」第三人称=玲奈子在邀玩"),
    671: ("甘织玲奈子", "回答妈妈，此场景无真唯/纱月"),
    672: ("玲奈子妈妈", "「真好啊～」=妈妈的接话"),
    682: ("王冢真唯", "真唯接送玲奈子，道谢「愿意答应我这种任性的请求」；遥奈不在场"),
    692: ("王冢真唯", "下句旁白「真唯有些害羞地垂下了眼帘」，「啊……嗯」是真唯"),
    743: ("王冢真唯", "真唯承诺「不会再做任何伤害你的事」，下句744玲奈子追问"),
    # —— 以下为对照原文逐段复核新发现、用户书签未覆盖（via=audit）——
    228: ("甘织玲奈子", "原文：香穗[227]「这种事交往前就知道了吧」后，玲奈子认输接话；下句旁白「不过她没有安慰我」证实[228]为玲奈子。话轮相位错一格"),
}

# 仅来自原文复核、不在用户书签内
AUDIT_LINES = {228}

# 复核存疑、原文无显式标记，建议听感确认（不计入 corrected_segments）
AUDIT_SUSPECTED = {
    280: ("小柳香穗", "甘织玲奈子",
          "香穗上句「果然啊～」已表示听懂，这句困惑提问更像玲奈子；原文无显式标记，待听感确认"),
}

# 书签标到、但经核对归属正确（疑似相邻叙述/边界误听，非分配错误）
FLAGGED_OK = {
    315: "「别再说了！」归真唯合理",
    379: "紫阳花道歉「总觉得很对不起你」，下句380玲奈子回应，归属正确",
    738: "「嗯，那个」存疑，可能真唯亦可能玲奈子犹豫，暂判归属可接受",
}


def main():
    L = load_lrc()
    seg = json.loads(SNAP.read_text(encoding="utf-8"))["segments"]
    assert len(L) == 750, len(L)

    data = json.loads(BM.read_text(encoding="utf-8"))
    sec = next(s for s in data["sections"] if s["fileName"] == "muli_5_part_001")
    bms = sorted(sec["manualBookmarks"], key=lambda b: b["positionMs"])

    manual_bookmarks = []
    for b in bms:
        pos = b["positionMs"] / 1000.0
        i = idx_at(L, pos)
        lo, hi = max(0, i - 2), min(len(L) - 1, i + 2)
        manual_bookmarks.append({
            "time": hhmmss(pos),
            "position_seconds": round(pos, 3),
            "label": b["label"],
            "matched_lrc_index": i,
            "matched_lrc": lrc_entry(L, i),
            "context": [lrc_entry(L, j) for j in range(lo, hi + 1)],
        })

    corrected = []
    for line in sorted(CORR):
        idx0 = line - 1
        s = seg[idx0]
        raw = {k: s.get(k) for k in RAW_FIELDS}
        expected, reason = CORR[line]
        corrected.append({
            "index": idx0,
            "via": "audit" if line in AUDIT_LINES else "bookmark",
            "raw_speaker": s["speaker"],
            "reviewed_speaker": s["speaker"],
            "expected_speaker": expected,
            "reason": reason,
            "raw": raw,
            "reviewed": dict(raw),
        })

    audit_suspected = [
        {"index": ln - 1, "raw_speaker": raw_sp, "suspected_speaker": exp,
         "reason": note, "text": seg[ln - 1]["text"]}
        for ln, (raw_sp, exp, note) in sorted(AUDIT_SUSPECTED.items())
    ]

    flagged_ok = [{"index": ln - 1, "speaker": seg[ln - 1]["speaker"],
                   "text": seg[ln - 1]["text"], "note": note}
                  for ln, note in sorted(FLAGGED_OK.items())]

    src_part = SRC_PART.read_text(encoding="utf-8") if SRC_PART.exists() else ""

    out = {
        "source": {
            "track_file": "muli_5_part_001",
            "volume": "第五卷",
            "bookmarks": str(BM),
            "lrc": str(LRC),
            "raw_snapshot": str(SNAP),
            "reviewed_snapshot": None,
            "original_full": str(SRC_FULL),
            "original_part001": str(SRC_PART),
            "sample_end_seconds": None,
            "coverage": "full_part",
            "note": "单快照：raw 即生成时实际使用的优化快照，expected_speaker 为人工判定的正确说话人",
        },
        "summary": {
            "manual_bookmarks": len(manual_bookmarks),
            "corrected_segments": len(corrected),
            "corrected_via_bookmark": sum(1 for c in corrected if c["via"] == "bookmark"),
            "corrected_via_audit": sum(1 for c in corrected if c["via"] == "audit"),
            "flagged_but_correct": len(flagged_ok),
            "audit_suspected": len(audit_suspected),
            "part_segments": 750,
            "original_part001_chars": len(src_part),
        },
        "sample_text": "\n".join(seg[i]["text"] for i in range(750)),
        "manual_bookmarks": manual_bookmarks,
        "corrected_segments": corrected,
        "flagged_but_correct": flagged_ok,
        "audit_suspected": audit_suspected,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  manual_bookmarks={len(manual_bookmarks)} corrected={len(corrected)} "
          f"flagged_ok={len(flagged_ok)}  total={len(corrected)+len(flagged_ok)}")


if __name__ == "__main__":
    main()
