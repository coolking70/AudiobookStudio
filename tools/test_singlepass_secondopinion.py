# -*- coding: utf-8 -*-
"""验证 audit_sample.py --singlepass 第二意见的召回（2026-06-14）。

在第五卷 part_001 上：把"单遍直出+角色表"的归因与流水线原始输出比对，分歧即 tier1 旗标
（suggest-only）。度量：① 40 处已知系统性错误里能旗到多少（recall）；② 旗标总量/精度权衡；
③ 断言流水线 speaker 全程未被改写（不破坏流水线）。需 AGNES_API_KEY。
"""
import json, os, sys, copy
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"BookVoiceParser"))
from book_voice_parser.single_pass_attributor import SinglePassAttributor
from book_voice_parser.batch_llm_attributor import BatchConfig
from book_voice_parser.schema import QuoteSpan

SNAP=Path(r"I:\code\aitts\text\task_snapshot_segments_2026-06-12_optimized.json")
seg=json.loads(SNAP.read_text(encoding="utf-8"))["segments"][:750]
full=(REPO/"docs/samples/第五卷/原文_part001.txt").read_text(encoding="utf-8")
KEY=os.environ.get("AGNES_API_KEY") or sys.exit("需 AGNES_API_KEY")
ROSTER={"甘织玲奈子":["玲奈子","小玲奈","甘织同学","甘织","丙方","玲奈子同学","小玲奈子","NEO·玲奈子"],
 "王冢真唯":["真唯","小真唯","王冢同学","真唯前辈","甲方"],"濑名紫阳花":["紫阳花","紫阳花同学","小紫","濑名","乙方"],
 "琴纱月":["纱月","纱月同学","小纱","小纱月"],"小柳香穗":["小香穗","香穗","小柳同学","小柳","香香"],
 "甘织遥奈":["遥奈","妹妹","小遥奈"],"高田卑弥呼":["高田同学","小卑弥","卑弥呼"],
 "照泽耀子":["照泽同学","耀子","耀子同学"],"羽贺铃兰":["羽贺同学","铃兰同学"],"玲奈子妈妈":["妈妈","母亲"]}
a2c={c:c for c in ROSTER}
for c,al in ROSTER.items():
    for x in al: a2c[x]=c
def canon(s):
    s=(s or "").strip()
    if s in a2c: return a2c[s]
    for x,c in a2c.items():
        if x and x in s: return c
    return s
CORR={187:"甘织玲奈子",281:"高田卑弥呼",310:"照泽耀子",311:"甘织玲奈子",314:"照泽耀子",318:"照泽耀子",319:"甘织玲奈子",324:"旁白",325:"照泽耀子",326:"甘织玲奈子",330:"旁白",334:"旁白",336:"旁白",361:"甘织玲奈子",362:"濑名紫阳花",367:"甘织玲奈子",389:"濑名紫阳花",398:"甘织玲奈子",400:"濑名紫阳花",428:"甘织玲奈子",429:"濑名紫阳花",431:"甘织玲奈子",455:"甘织玲奈子",465:"甘织玲奈子",473:"甘织遥奈",486:"甘织玲奈子",490:"甘织遥奈",506:"旁白",511:"甘织遥奈",514:"甘织遥奈",530:"甘织玲奈子",556:"甘织玲奈子",573:"甘织玲奈子",590:"琴纱月",604:"琴纱月",651:"琴纱月",661:"甘织玲奈子",671:"甘织玲奈子",672:"玲奈子妈妈",682:"王冢真唯",692:"王冢真唯",743:"王冢真唯",228:"甘织玲奈子",554:"甘织玲奈子",558:"甘织玲奈子",560:"甘织玲奈子"}
SCENES=[(305,336),(358,400),(425,456),(461,532),(553,605),(645,675),(215,229)]
def is_d(s): return s.get("attribution_type")!="narrator" and s["speaker"]!="旁白"
testlines=[]; quotes=[]
for a,b in SCENES:
    for i in range(a-1,b):
        if not is_d(seg[i]): continue
        ln=i+1; qid=seg[i]["quote_id"] or f"L{ln}"
        testlines.append((ln,qid)); 
        quotes.append(QuoteSpan(quote_id=qid,text=seg[i]["text"],start=0,end=len(seg[i]["text"]),context_before="",context_after=""))
qid2ln={qid:ln for ln,qid in testlines}
known_err=[ln for ln,_ in testlines if ln in CORR]

snap_before=copy.deepcopy(seg)  # 验证不被改写
cfg=BatchConfig(base_url="https://apihub.agnes-ai.com/v1",api_key=KEY,model="agnes-2.0-flash",timeout=300)
sp=SinglePassAttributor(cfg,full_text=full,chunk_size=200)
ramap={c:[x for x in al if x!=c] for c,al in ROSTER.items()}
res=sp.attribute(quotes,role_hints=list(ROSTER.keys()),narrator="甘织玲奈子",role_aliases=ramap)
sppred={qid2ln[q]:canon(a.speaker) for q,a in res.items() if q in qid2ln}

# tier1 旗标 = 单遍 ≠ 流水线（suggest-only）
flagged=[ln for ln,_ in testlines if ln in sppred and sppred[ln]!=canon(seg[ln-1]["speaker"])]
caught=[ln for ln in known_err if ln in flagged]
missed=[ln for ln in known_err if ln not in flagged]
# precision 下界：旗标里命中已知错误的比例（未知错误会被算作"误报"，故为下界）
tp=len(caught); fp=len(flagged)-tp
print(f"评测集对话行 {len(testlines)}，已知系统性错误 {len(known_err)}")
print(f"单遍第二意见旗标(tier1) 共 {len(flagged)} 段")
print(f"  召回 recall = {tp}/{len(known_err)} = {tp/len(known_err):.0%}（旗到的已知错误）")
print(f"  精度下界 precision≥ {tp}/{len(flagged)} = {tp/max(1,len(flagged)):.0%}（其余多为双人轮换相位翻转/未知错误）")
print(f"  漏旗的已知错误: {missed}")
# 不破坏流水线
unchanged = all(snap_before[i]["speaker"]==seg[i]["speaker"] for i in range(750))
print(f"流水线 speaker 全程未改写（suggest-only）: {unchanged}")
sys.exit(0 if (tp/len(known_err)>=0.8 and unchanged) else 1)
