"""Analyze whether structured baseline evidence helps review routing.

Reads parse JSON files plus the authoritative groundtruth samples and reports:
- scoring by segment, using the same text/crowd/canonicalization rules as run_regression
- structured evidence coverage (E=...;R=...;S=...)
- accuracy grouped by evidence and risk tags
- rough review-routing recall/precision for low confidence or risk tags
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "tools"))

from run_regression import MANIFEST, canon, is_crowd, norm, pick_segs  # noqa: E402


STRUCTURED_RE = re.compile(
    r"(?:^|[；;\s])E=(?P<e>[^;；\n]+)[;；]R=(?P<r>[^;；\n]+)(?:[;；]S=(?P<s>[^;；\n]+))?",
)


def parse_tags(evidence: str) -> tuple[list[str], list[str], str]:
    match = STRUCTURED_RE.search(str(evidence or ""))
    if not match:
        return [], [], ""
    e_tags = [x.strip() for x in match.group("e").split(",") if x.strip()]
    r_tags = [x.strip() for x in match.group("r").split(",") if x.strip() and x.strip() != "none"]
    signal = (match.group("s") or "").strip()
    return e_tags, r_tags, signal


def accuracy(pair: list[int]) -> str:
    return f"{pair[0]}/{pair[1]}={pair[0] / pair[1]:.1%}" if pair[1] else "—"


def add_bucket(buckets: dict[str, list[int]], key: str, ok: bool) -> None:
    bucket = buckets.setdefault(key, [0, 0])
    bucket[0] += int(ok)
    bucket[1] += 1


def load_parse(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments", data if isinstance(data, list) else [])


def analyze_sample(seg: str, parse_path: Path, rows: list[dict[str, Any]]) -> None:
    gt = json.loads((SAMP / f"{seg}_groundtruth.json").read_text(encoding="utf-8"))
    parse = load_parse(parse_path)
    for g in gt["segments"]:
        i = int(g["i"])
        if is_crowd(g["speaker"]):
            continue
        if i >= len(parse) or norm(parse[i].get("text", "")) != norm(g["text"]):
            continue
        p = parse[i]
        acceptable = {canon(x) for x in (g.get("acceptable") or [g["speaker"]])}
        acceptable.add(canon(g["speaker"]))
        got = canon(str(p.get("speaker") or ""))
        ok = got in acceptable
        evidence = str(p.get("evidence") or p.get("_evidence") or "")
        e_tags, r_tags, signal = parse_tags(evidence)
        try:
            confidence = float(p.get("confidence"))
        except (TypeError, ValueError):
            confidence = 1.0
        rows.append({
            "seg": seg,
            "i": i,
            "ok": ok,
            "speaker": got,
            "truth": canon(str(g["speaker"])),
            "confidence": confidence,
            "evidence_tags": e_tags,
            "risk_tags": r_tags,
            "signal": signal,
            "structured": bool(e_tags or r_tags),
        })


def print_group(title: str, buckets: dict[str, list[int]], limit: int = 20) -> None:
    print(f"\n== {title} ==")
    for key, pair in sorted(buckets.items(), key=lambda kv: (-kv[1][1], kv[0]))[:limit]:
        print(f"{key:<28} {accuracy(pair):>16}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-dir", required=True, help="Directory containing <seg>_parse.json")
    ap.add_argument("--only", help="Comma-separated sample subset, e.g. seg8,seg9")
    ap.add_argument("--risk-tags", default="multi_speaker,addressee_trap,weak_turn,generic_relation,alias_ambiguous,off_scene")
    ap.add_argument("--confidence-threshold", type=float, default=0.7)
    args = ap.parse_args()

    parse_dir = Path(args.parse_dir)
    rows: list[dict[str, Any]] = []
    for seg in pick_segs(args.only):
        parse_path = parse_dir / f"{seg}_parse.json"
        if parse_path.exists():
            analyze_sample(seg, parse_path, rows)

    if not rows:
        raise SystemExit(f"No scored rows found in {parse_dir}")

    correct = sum(1 for r in rows if r["ok"])
    structured = sum(1 for r in rows if r["structured"])
    print(f"样本数: {len({r['seg'] for r in rows})}  具名段: {len(rows)}  准确率: {correct / len(rows):.2%}")
    print(f"结构化依据覆盖: {structured}/{len(rows)} = {structured / len(rows):.1%}")

    by_seg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_evidence: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_risk: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        add_bucket(by_seg, r["seg"], r["ok"])
        for tag in r["evidence_tags"] or ["<missing>"]:
            add_bucket(by_evidence, tag, r["ok"])
        for tag in r["risk_tags"] or ["none"]:
            add_bucket(by_risk, tag, r["ok"])
    print_group("按样本", by_seg)
    print_group("按证据标签 E", by_evidence)
    print_group("按风险标签 R", by_risk)

    risk_set = {x.strip() for x in args.risk_tags.split(",") if x.strip()}
    flagged = [
        r for r in rows
        if r["confidence"] < args.confidence_threshold or bool(set(r["risk_tags"]) & risk_set)
    ]
    errors = [r for r in rows if not r["ok"]]
    flagged_errors = [r for r in flagged if not r["ok"]]
    print("\n== 复核路由粗评 ==")
    print(f"标记段: {len(flagged)}/{len(rows)} = {len(flagged) / len(rows):.1%}")
    print(f"错误召回: {len(flagged_errors)}/{len(errors)} = {(len(flagged_errors) / len(errors)) if errors else 0:.1%}")
    print(f"标记精度: {len(flagged_errors)}/{len(flagged)} = {(len(flagged_errors) / len(flagged)) if flagged else 0:.1%}")

    misses = [r for r in errors if r not in flagged]
    if misses:
        print("\n未被证据/低置信标记捕获的错误（前 12 条）:")
        for r in misses[:12]:
            print(f"- {r['seg']}#{r['i']}: got={r['speaker']} truth={r['truth']} "
                  f"conf={r['confidence']:.2f} E={','.join(r['evidence_tags']) or '-'} "
                  f"R={','.join(r['risk_tags']) or '-'} S={r['signal'] or '-'}")


if __name__ == "__main__":
    main()
