"""Deterministically build a sample's groundtruth.json from its parse + review.

固化（finalize）一个样本：把人工复核得到的 `*_review.json` 修正叠加到模型解析
`*_parse.json` 上，算出具名准确率等口径，产出权威的 `*_groundtruth.json`，并顺便
重新生成与解析一一对应的 `*_transcript.txt`（模型口径的人类可读视图）。

所有字段都是从 parse + review 机械推导出来的——没有任何手工计数，所以结果可复现、
可被 verify_sample.py 校验。

用法（在 seg5 上等价于重新固化一遍）：
    .venv/bin/python tools/build_groundtruth.py --seg muli4_seg5
或显式指定路径：
    .venv/bin/python tools/build_groundtruth.py \
        --parse docs/samples/X_parse.json \
        --review docs/samples/X_review.json \
        --out    docs/samples/X_groundtruth.json

口径定义（与 tools/eval_external_model.py 一致）：
- crowd（群众/不计分）  : speaker 以「群众·」「厕所女生」开头，或属于
                          {未知, 未知临时人物, 旁白, 其他, ""}
- crowd_segments        : 终值(corrected)说话人为 crowd 的段数
- named_total           : total_segments - crowd_segments
- named_corrections     : 计入 named_total 的段里，终值≠模型值 的段数（即模型答错的具名句）
- model_named_accuracy  : (named_total - named_corrections) / named_total
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"

PARSER_LABEL = "agnes-2.0-flash + block_review + dense_scene_review_routing + address_term_backcheck"
CROWD_PREFIXES = ("群众·", "厕所女生")
CROWD_LITERALS = {"未知", "未知临时人物", "旁白", "其他", ""}


def is_crowd(speaker: str) -> bool:
    s = speaker or ""
    return s.startswith(CROWD_PREFIXES) or s in CROWD_LITERALS


def build(parse_path: Path, review_path: Path) -> tuple[dict, str]:
    parse = json.loads(parse_path.read_text(encoding="utf-8"))
    psegs = parse["segments"]
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    corrections: dict[str, str] = review.get("corrections", {})

    # the raw_file the parse was built from — try to read it back out of nothing; we
    # only record the conventional name here (caller keeps them named together).
    stem = parse_path.name.replace("_parse.json", "")

    gt_segments = []
    transcript_lines = []
    crowd_segments = 0
    named_total = 0
    named_corrections = 0
    corrected_indices = []

    for i, p in enumerate(psegs):
        model_speaker = p.get("speaker", "")
        corr = corrections.get(str(i))
        # review value "A/B" = 两人皆可（acceptable-set），主说话人取第一个
        acceptable = [a.strip() for a in corr.split("/")] if corr is not None else None
        final = acceptable[0] if acceptable else model_speaker
        corrected = corr is not None and final != model_speaker
        if corrected:
            corrected_indices.append(i)

        seg = {
            "i": i,
            "speaker": final,
            "model_speaker": model_speaker,
            "text": p.get("text", ""),
            "corrected": corrected,
        }
        if acceptable and len(acceptable) > 1:
            seg["acceptable"] = acceptable
        gt_segments.append(seg)

        # accuracy is scored on the FINAL (human-authoritative) speaker；
        # 模型命中 acceptable 集合内任一即算对
        ok_set = set(acceptable) if acceptable else {final}
        if is_crowd(final):
            crowd_segments += 1
        else:
            named_total += 1
            if model_speaker not in ok_set:
                named_corrections += 1

        # transcript renders the MODEL speaker (what the parse said, pre-review)
        conf = p.get("confidence")
        atype = p.get("attribution_type") or ""
        transcript_lines.append(f"[{i:>3}] {model_speaker:<9}({conf}) {atype}".rstrip())
        transcript_lines.append(f"      「{p.get('text', '')}」")

    accuracy = round((named_total - named_corrections) / named_total, 4) if named_total else 0.0

    groundtruth = {
        "source": {
            "raw_file": f"docs/samples/{stem}_sample.txt",
            "parse_file": f"docs/samples/{parse_path.name}",
            "parser": PARSER_LABEL,
            "finalized": datetime.now().isoformat(timespec="seconds"),
            "human_reviewed": True,
            "crowd_convention": "群众·<群体名>，评分按 crowd:true 排除",
        },
        "total_segments": len(psegs),
        "crowd_segments": crowd_segments,
        "named_corrections": named_corrections,
        "named_total": named_total,
        "model_named_accuracy": accuracy,
        "corrected_indices": corrected_indices,
        "segments": gt_segments,
    }
    return groundtruth, "\n".join(transcript_lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", help="样本前缀，如 muli4_seg5（自动推导 parse/review/out 路径）")
    ap.add_argument("--parse", type=Path)
    ap.add_argument("--review", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--transcript", type=Path, help="同时重写 transcript.txt（默认按 --seg 推导）")
    ap.add_argument("--no-transcript", action="store_true", help="不重写 transcript.txt")
    args = ap.parse_args()

    if args.seg:
        parse = args.parse or SAMP / f"{args.seg}_parse.json"
        review = args.review or SAMP / f"{args.seg}_review.json"
        out = args.out or SAMP / f"{args.seg}_groundtruth.json"
        transcript = args.transcript or SAMP / f"{args.seg}_transcript.txt"
    else:
        if not (args.parse and args.out):
            raise SystemExit("请给出 --seg，或同时给出 --parse 和 --out")
        parse, review, out = args.parse, args.review or Path("/nonexistent"), args.out
        transcript = args.transcript

    groundtruth, transcript_text = build(parse, review)

    # 重建对照基线（reference_baseline）样本时，保留其基线标记与诚实的 parser 标签，
    # 不让流水线默认标签覆盖掉。
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        if prev.get("reference_baseline"):
            groundtruth["reference_baseline"] = True
            if "reference_note" in prev:
                groundtruth["reference_note"] = prev["reference_note"]
            if prev.get("source", {}).get("parser"):
                groundtruth["source"]["parser"] = prev["source"]["parser"]

    out.write_text(json.dumps(groundtruth, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_transcript and transcript:
        transcript.write_text(transcript_text, encoding="utf-8")

    print(f"✓ 写出 {out.name}")
    print(f"  总段数 {groundtruth['total_segments']} · 群众 {groundtruth['crowd_segments']} · "
          f"具名 {groundtruth['named_total']}")
    print(f"  修正 {groundtruth['named_corrections']} 处 → 模型具名准确率 "
          f"{groundtruth['model_named_accuracy']:.2%}")
    print(f"  修正段索引: {groundtruth['corrected_indices']}")
    if not args.no_transcript and transcript:
        print(f"✓ 重写 {transcript.name}")


if __name__ == "__main__":
    main()
