"""Acceptance-check a固化好的样本：交叉校验 parse / review / groundtruth 三者一致。

这是「验收」一个 seg 的标准脚本——把人工复核固化的结果跑一遍机器校验，0 错误才算合格。
新样本由其他 AI 产出后，先跑这个再提交。

用法：
    .venv/bin/python tools/verify_sample.py --seg muli4_seg5
    .venv/bin/python tools/verify_sample.py --all     # 校验 docs/samples 下所有 seg

校验项：
  1. parse 段数 == groundtruth 段数
  2. 每段 groundtruth.text == parse.text，groundtruth.model_speaker == parse.speaker
  3. review 里每条修正都已落到 groundtruth（且 corrected=True、终值正确）
  4. 未修正段满足 speaker == model_speaker 且 corrected=False
  5. corrected_indices 字段 == 实际 corrected=True 的段
  6. crowd_segments / named_total / named_corrections / model_named_accuracy 口径自洽
  7. 流水线溯源：parse 是 BookVoiceParser 完整流水线产物（stats 含 block_review 等阶段、
     attribution_type 为小写 JSON 值、段含 quote_id/candidates 等字段），不是裸模型直出。
     groundtruth 标了 reference_baseline=true 的对照基线豁免此项（降级为提示）。
  8. sample.txt 存在；transcript.txt 与最终 parse 的模型口径一致（仅告警，不算失败）
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"

CROWD_PREFIXES = ("群众·", "厕所女生")
CROWD_LITERALS = {"未知", "未知临时人物", "旁白", "其他", ""}


def is_crowd(s: str) -> bool:
    s = s or ""
    return s.startswith(CROWD_PREFIXES) or s in CROWD_LITERALS


def verify(seg: str) -> tuple[list[str], list[str]]:
    errs: list[str] = []
    warns: list[str] = []

    parse_p = SAMP / f"{seg}_parse.json"
    review_p = SAMP / f"{seg}_review.json"
    gt_p = SAMP / f"{seg}_groundtruth.json"
    sample_p = SAMP / f"{seg}_sample.txt"
    transcript_p = SAMP / f"{seg}_transcript.txt"

    for p in (parse_p, gt_p):
        if not p.exists():
            return [f"缺少必需文件 {p.name}"], warns

    parse = json.loads(parse_p.read_text(encoding="utf-8"))
    psegs = parse["segments"]
    gt = json.loads(gt_p.read_text(encoding="utf-8"))
    gsegs = gt["segments"]
    corr = json.loads(review_p.read_text(encoding="utf-8")).get("corrections", {}) if review_p.exists() else {}

    # 1. counts
    if len(psegs) != len(gsegs):
        errs.append(f"段数不一致 parse={len(psegs)} groundtruth={len(gsegs)}")
        return errs, warns

    flagged = []
    for g in gsegs:
        i = g["i"]
        p = psegs[i]
        # 2. text & model_speaker alignment
        if g["text"] != p.get("text", ""):
            errs.append(f"i={i} text 与 parse 不一致")
        if g["model_speaker"] != p.get("speaker", ""):
            errs.append(f"i={i} model_speaker={g['model_speaker']} != parse.speaker={p.get('speaker')}")
        si = str(i)
        if si in corr:
            # 3. correction applied（"A/B" = acceptable-set，主说话人取第一个）
            acceptable = [a.strip() for a in corr[si].split("/")]
            if g["speaker"] != acceptable[0]:
                errs.append(f"i={i} 修正未落地: gt={g['speaker']} review={corr[si]}")
            if len(acceptable) > 1 and g.get("acceptable") != acceptable:
                errs.append(f"i={i} acceptable 集合未写入: gt={g.get('acceptable')} review={corr[si]}")
            if acceptable[0] != g["model_speaker"] and not g.get("corrected"):
                errs.append(f"i={i} 应标 corrected=True")
        else:
            # 4. untouched segment
            if g["speaker"] != g["model_speaker"]:
                errs.append(f"i={i} 无修正但 speaker!=model_speaker")
            if g.get("corrected"):
                errs.append(f"i={i} 标了 corrected=True 但 review 里没有")
        if g.get("corrected"):
            flagged.append(i)

    # review 里指向越界/不存在段的修正
    for si in corr:
        if not (0 <= int(si) < len(gsegs)):
            errs.append(f"review 修正 i={si} 超出段范围")

    # 5. corrected_indices field
    if gt.get("corrected_indices") != flagged:
        errs.append(f"corrected_indices 字段={gt.get('corrected_indices')} != 实际={flagged}")

    # 6. metrics
    crowd = sum(1 for g in gsegs if is_crowd(g["speaker"]))
    named = [g for g in gsegs if not is_crowd(g["speaker"])]
    named_corr = sum(1 for g in named
                     if g["model_speaker"] not in set(g.get("acceptable") or [g["speaker"]]))
    acc = round((len(named) - named_corr) / len(named), 4) if named else 0.0
    checks = {
        "total_segments": (gt.get("total_segments"), len(gsegs)),
        "crowd_segments": (gt.get("crowd_segments"), crowd),
        "named_total": (gt.get("named_total"), len(named)),
        "named_corrections": (gt.get("named_corrections"), named_corr),
        "model_named_accuracy": (gt.get("model_named_accuracy"), acc),
    }
    for name, (got, exp) in checks.items():
        if got != exp:
            errs.append(f"口径 {name} 字段={got} 计算={exp}")

    # 7. 流水线溯源：parse 必须是 BookVoiceParser 完整流水线产物，而非裸模型单遍直出。
    # reference_baseline=true 的对照样本豁免（降级为提示），但仍记录证据。
    is_ref = bool(gt.get("reference_baseline"))
    prov = []
    stats = parse.get("stats") or {}
    REQUIRED_STAGES = ("block_review", "address_term_backcheck", "scene_state")
    missing = [s for s in REQUIRED_STAGES if s not in stats]
    if missing:
        prov.append(f"parse.stats 缺少流水线阶段 {missing}（疑似未跑完整 BookVoiceParser 流水线）")
    # 枚举被错误序列化成 repr（如 "AttributionType.IMPLICIT"）而非 JSON 值（"implicit"）
    bad_enum = [p.get("attribution_type") for p in psegs
                if isinstance(p.get("attribution_type"), str) and "." in p.get("attribution_type")]
    if bad_enum:
        prov.append(f"attribution_type 为枚举 repr（如 {bad_enum[0]}），应是小写 JSON 值——序列化路径不对")
    PIPELINE_FIELDS = ("quote_id", "candidates", "scene_characters")
    if psegs and not any(f in psegs[0] for f in PIPELINE_FIELDS):
        prov.append(f"parse 段缺少流水线字段 {PIPELINE_FIELDS}（疑似简化直出）")

    if prov:
        if is_ref:
            warns.append("【对照基线】非流水线产物，已豁免溯源校验：" + "；".join(prov))
        else:
            errs.extend("流水线溯源：" + m for m in prov)

    # 8. companion files (warnings only)
    if not sample_p.exists():
        warns.append(f"缺少 {sample_p.name}（评测可跑，但无法溯源原文切片）")
    if transcript_p.exists():
        tlines = transcript_p.read_text(encoding="utf-8").splitlines()
        n_head = sum(1 for l in tlines if l.startswith("["))
        if n_head != len(psegs):
            warns.append(f"transcript 行数对应 {n_head} 段，与 parse {len(psegs)} 段不符（视图过时）")

    return errs, warns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", help="样本前缀，如 muli4_seg5")
    ap.add_argument("--all", action="store_true", help="校验所有 *_groundtruth.json")
    args = ap.parse_args()

    if args.all:
        segs = sorted(Path(p).name.replace("_groundtruth.json", "")
                      for p in glob.glob(str(SAMP / "*_groundtruth.json")))
    elif args.seg:
        segs = [args.seg]
    else:
        raise SystemExit("请给出 --seg <前缀> 或 --all")

    any_err = False
    for seg in segs:
        errs, warns = verify(seg)
        status = "✓ PASS" if not errs else "✗ FAIL"
        print(f"{status}  {seg}")
        for w in warns:
            print(f"   ⚠ {w}")
        for e in errs:
            print(f"   ✗ {e}")
            any_err = True
    sys.exit(1 if any_err else 0)


if __name__ == "__main__":
    main()
