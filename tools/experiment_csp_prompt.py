"""A/B the block-decoder prompt: current ("turn-order") vs CSP-style
("converge active participants -> fingerprint/adjacency anchoring -> elimination").

Isolated: one underlying parse per sample (block OFF), then apply block review with
each prompt on a deep copy. Scored against authoritative GT for both samples; the seg2
4-person opening is reported explicitly.
"""
from __future__ import annotations
import copy, json, os, re, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "BookVoiceParser"))
sys.path.insert(0, str(REPO / "tools"))
from book_voice_parser import BatchConfig, parse_novel  # noqa
from book_voice_parser.block_review import apply_block_review  # noqa
from book_voice_parser.cleaner import normalize_text  # noqa
from book_voice_parser.quote_extractor import extract_quotes  # noqa
from book_voice_parser.alias_registry import AliasRegistry  # noqa
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa

a2c = {}
for c, al in ROLE_HINTS.items():
    if isinstance(al, (list, tuple)):
        for a in al:
            a2c[a] = c
canon = lambda x: a2c.get(x, x)
SIS = {"甘织玲奈子", "甘织遥奈"}
CROWD = lambda x: (x or "").startswith("厕所女生") or x in {"未知", "未知临时人物", "国中玲奈子", "旁白", "其他", ""}


def norm(s):
    s = re.sub(r"[「」『』]", "", str(s or ""))
    return "".join(s.split()).replace("彷佛", "仿佛").replace("姊", "姐")


CSP_PROMPT = """下面是一段连续的小说原文（旁白与对话混排），每句对话前有【{prefix}NNNN】标记。

角色表（只用其中的规范名，不要用别名）：
{roster}
叙述者（第一人称“我”）：{narrator}

请像侦探一样，按【约束满足】的方式判断说话人，不要逐句孤立地猜：

第一步·确定本段【实际参与对话的人】：
- 通读全段。用点名、感谢、招呼、排除等线索收窄范围——例如有人说“谢谢A、B、C”，说明被点到的人在场且在参与；在场但全程既没被提及也没接话的人，很可能没参与这段对话，应排除在外。
- 在心里列出这段真正在对话的小名单（可能少于全部在场角色）。

第二步·先锚定能确定的句子：
- 旁白里的“X说道/X笑着/X问”等显式标记，直接锚定该句说话人。
- 称呼风格：每个角色称呼别人的方式往往固定（有人惯用姓氏，有人用昵称/亲昵称呼）。某句台词里出现的称呼风格可以反推说话人是谁——这是软证据，可被更强证据推翻，但在缺乏其它线索时很有用。

第三步·用【邻接对】+【语气性格】+【排除法】补齐其余句：
- 被点名或被提问的人，常常就是下一句的说话人。
- 提问→回答、提议→拒绝或接受、递出→接住，多为两人交替；同一人连续做一件事可为续说。
- 角色性格语气要吻合（强势/怯懦/毒舌/温柔/冷淡）。
- 在第一步的小名单内，结合已锚定的句子用排除法定剩余句；若仍没有把握，宁可保留原判也不要乱猜。

补充：第一人称“我”说出口的话归{narrator}；『』内心独白仍归发出该想法者；无名路人归“其他”；不是台词的归“旁白”。

【本段原文】：
{block}

只输出一个 JSON 对象：键为本段每个编号，值为规范名。例如 {{"{prefix}0012":"角色甲","{prefix}0013":"角色乙"}}。不要任何解释。"""


def cfg():
    return BatchConfig(base_url="https://apihub.agnes-ai.com/v1", api_key=os.environ["AGNES_API_KEY"],
                       model="agnes-2.0-flash", batch_size=8, max_tokens=5000, temperature=0.0,
                       timeout=180, context_chars=320, output_mode="compact", disable_thinking=True)


def parse_off(raw):
    res = parse_novel(raw, role_hints=ROLE_HINTS, batch_llm_config=cfg(), narrator="甘织玲奈子",
                      return_result=True, include_narration=False, review_threshold=0.7,
                      enable_block_review=False)
    cleaned = normalize_text(raw)
    quotes = extract_quotes(cleaned)
    return res.segments, quotes, cleaned


def review(off, quotes, cleaned, prompt):
    segs = copy.deepcopy(off)
    aliases = AliasRegistry.from_role_hints(ROLE_HINTS)
    apply_block_review(segs, quotes, cleaned, cfg(), narrator="甘织玲奈子",
                       role_hints=list(ROLE_HINTS.keys()), aliases=aliases, prompt_template=prompt)
    return segs


def s1_gt():
    snap = json.loads((SAMP / "task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json").read_text(encoding="utf-8"))["segments"]
    reg = json.loads((SAMP / "muli4_part001_first_hour_bookmark_regression.json").read_text(encoding="utf-8"))
    gt = {}
    for s in snap:
        sp = str(s.get("speaker") or ""); t = norm(s.get("text", ""))
        if sp and sp != "旁白" and t:
            gt.setdefault(t, sp)
    for c in reg["corrected_segments"]:
        gt[norm(snap[int(c["index"])].get("text", ""))] = str(c["expected_speaker"])
    cnt = Counter(norm(s.get("text", "")) for s in snap if s.get("speaker") and s.get("speaker") != "旁白")
    return gt, cnt


def score_text(segs, gt, cnt):
    pc = Counter(norm(s.text) for s in segs)
    nc = nt = sc = st = 0
    for s in segs:
        k = norm(s.text)
        if k not in gt:
            continue
        g = canon(gt[k]); p = canon(s.speaker)
        if g in SIS: st += 1; sc += int(g == p)
        if pc[k] == 1 and cnt.get(k, 0) == 1 and not CROWD(g):
            nt += 1; nc += int(g == p)
    return nc, nt, sc, st


def score_index(segs, gtsegs):
    nc = nt = sc = st = 0
    n = min(len(segs), len(gtsegs))
    for i in range(n):
        if norm(segs[i].text) != norm(gtsegs[i]["text"]):
            continue
        g = canon(gtsegs[i]["speaker"]); p = canon(segs[i].speaker)
        if g in SIS: st += 1; sc += int(g == p)
        if not CROWD(g):
            nt += 1; nc += int(g == p)
    return nc, nt, sc, st


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")

    print("=== 样本1 first-hour ===", flush=True)
    off1, q1, c1 = parse_off((lambda t: t[t.find("第四卷 序章"):t.find("是为了欢迎我回来", t.find("第四卷 序章"))+len("是为了欢迎我回来")+80].strip())((SAMP/"muli4_original_125697_utf8.txt").read_text(encoding="utf-16")))
    gt1, cnt1 = s1_gt()
    for label, prompt in [("当前prompt", None), ("CSP prompt", CSP_PROMPT)]:
        nc, nt, sc, st = score_text(review(off1, q1, c1, prompt), gt1, cnt1)
        print(f"  {label}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {sc}/{st}={sc/max(1,st):.1%}", flush=True)

    print("\n=== 样本2 seg2 ===", flush=True)
    off2, q2, c2 = parse_off((SAMP/"muli4_seg2_sample.txt").read_text(encoding="utf-8"))
    gt2 = json.loads((SAMP/"muli4_seg2_groundtruth.json").read_text(encoding="utf-8"))["segments"]
    for label, prompt in [("当前prompt", None), ("CSP prompt", CSP_PROMPT)]:
        segs = review(off2, q2, c2, prompt)
        nc, nt, sc, st = score_index(segs, gt2)
        opening = []
        for j in range(4):
            g = canon(gt2[j]["speaker"]); p = canon(segs[j].speaker)
            opening.append(("✅" if g == p else "❌") + f"[{j}]真值{gt2[j]['speaker']}/解析{segs[j].speaker}")
        print(f"  {label}: 具名 {nc}/{nt}={nc/max(1,nt):.1%}  姐妹 {sc}/{st}={sc/max(1,st):.1%}", flush=True)
        print(f"     开场: {'  '.join(opening)}", flush=True)


if __name__ == "__main__":
    main()
