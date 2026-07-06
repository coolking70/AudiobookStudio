"""Review structured Agnes baseline evidence with a second model.

This experiment asks the reviewer to verify/refute the baseline evidence
(`E=...;R=...;S=...`) rather than simply re-guessing the speaker.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "tools"))

from run_regression import canon, is_crowd, norm, score_sample  # noqa: E402


PROVIDERS = {
    "tokenhub": "https://tokenhub.tencentmaas.com/v1",
    "agnes": "https://apihub.agnes-ai.com/v1",
    "sensenova": "https://token.sensenova.cn/v1",
}


STRUCTURED_RE = re.compile(
    r"(?:^|[；;\s])E=(?P<e>[^;；\n]+)[;；]R=(?P<r>[^;；\n]+)(?:[;；]S=(?P<s>[^;；\n]+))?",
)

STRONG_REASON_RE = re.compile(
    r"明示|明确|证实|台词第一人称|第一人称|屏幕|被拉|语气.*喵|喵|伸手邀请|惊讶回应|嘟囔"
)


SYSTEM_PROMPT = """\
你是中文小说有声书说话人复核专家。

任务不是重新猜一次，而是验证/反驳基线归因的结构化证据：
- 若基线证据 E/R/S 与原文一致，且没有更强反证，decision=keep。
- 若基线证据不成立，或存在更强原文证据指向另一位说话人，decision=revise。
- 台词中被称呼的人通常是受话人，不是说话人。
- 仅凭对话轮换/上一句推测属于弱证据；多人同场时要更保守。
- 不要输出解释性正文，只输出 JSON 数组。
- 如果你认为 speaker 不需要改变，decision 必须为 keep，不要返回 revise。
- revise 只用于改成与当前 speaker 不同的角色。

返回每个目标：
{"index":整数,"decision":"keep|revise|uncertain","speaker":"角色名","confidence":0.0到1.0,"reason":"≤24字，指出证据成立或反证"}
"""


def parse_evidence(evidence: str) -> dict[str, str]:
    m = STRUCTURED_RE.search(str(evidence or ""))
    if not m:
        return {"E": "", "R": "", "S": ""}
    return {"E": m.group("e") or "", "R": m.group("r") or "", "S": m.group("s") or ""}


def load_segments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments", data if isinstance(data, list) else [])


def load_truth(seg: str) -> list[dict[str, Any]]:
    return json.loads((SAMP / f"{seg}_groundtruth.json").read_text(encoding="utf-8"))["segments"]


def scored_rows(seg: str, parse_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for g in load_truth(seg):
        i = int(g["i"])
        if is_crowd(g["speaker"]) or i >= len(parse_segments):
            continue
        p = parse_segments[i]
        if norm(p.get("text", "")) != norm(g["text"]):
            continue
        acceptable = {canon(x) for x in (g.get("acceptable") or [g["speaker"]])}
        acceptable.add(canon(g["speaker"]))
        got = canon(str(p.get("speaker") or ""))
        rows.append({
            "i": i,
            "ok": got in acceptable,
            "truth": canon(str(g["speaker"])),
            "speaker": got,
        })
    return rows


def has_risk(seg: dict[str, Any]) -> bool:
    ev = str(seg.get("evidence") or "")
    tags = parse_evidence(ev)
    risks = {x.strip() for x in tags["R"].split(",") if x.strip() and x.strip() != "none"}
    try:
        conf = float(seg.get("confidence"))
    except (TypeError, ValueError):
        conf = 1.0
    return conf < 0.7 or bool(risks)


def weak_evidence(seg: dict[str, Any]) -> bool:
    tags = parse_evidence(str(seg.get("evidence") or ""))
    e_tags = {x.strip() for x in tags["E"].split(",") if x.strip()}
    weak = {"dialogue_alternation", "narrator_anchor", "scene_presence", "address_term", "unknown_weak"}
    return bool(e_tags & weak)


def load_audit(path: Path | None) -> dict[int, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments") or {}
    out: dict[int, dict[str, Any]] = {}
    for key, value in segments.items():
        try:
            out[int(key)] = value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            continue
    return out


def audit_disagrees(parse_seg: dict[str, Any], audit_item: dict[str, Any]) -> bool:
    reask = str(audit_item.get("reask") or "").strip()
    if not reask:
        return False
    current = str(parse_seg.get("speaker") or "").strip()
    return bool(reask) and canon(reask) != canon(current)


def choose_targets(
    seg_name: str,
    parse_segments: list[dict[str, Any]],
    mode: str,
    limit: int | None,
    audit: dict[int, dict[str, Any]] | None = None,
) -> list[int]:
    rows = scored_rows(seg_name, parse_segments)
    by_i = {r["i"]: r for r in rows}
    if mode == "errors":
        indices = [r["i"] for r in rows if not r["ok"]]
    elif mode == "errors-and-controls":
        errors = [r["i"] for r in rows if not r["ok"]]
        controls = [r["i"] for r in rows if r["ok"] and has_risk(parse_segments[r["i"]])]
        indices = errors + controls[: max(len(errors), 12)]
    elif mode == "audit-disagree":
        audit = audit or {}
        indices = [
            i for i in sorted(by_i)
            if i in audit
            and audit_disagrees(parse_segments[i], audit[i])
            and (has_risk(parse_segments[i]) or weak_evidence(parse_segments[i]))
        ]
    elif mode == "audit-disagree-all":
        audit = audit or {}
        indices = [
            i for i in sorted(by_i)
            if i in audit and audit_disagrees(parse_segments[i], audit[i])
        ]
    else:
        indices = [i for i in sorted(by_i) if has_risk(parse_segments[i])]
    return indices[:limit] if limit else indices


def build_user_prompt(
    seg_name: str,
    parse_segments: list[dict[str, Any]],
    indices: list[int],
    audit: dict[int, dict[str, Any]] | None = None,
) -> str:
    lines = [f"样本：{seg_name}", "请复核以下目标："]
    for i in indices:
        s = parse_segments[i]
        ev = parse_evidence(str(s.get("evidence") or ""))
        candidates = "、".join(str(x) for x in (s.get("candidates") or [])[:16])
        audit_item = (audit or {}).get(i) or {}
        audit_line = ""
        if audit_item.get("reask"):
            audit_line = f"机器审计第二意见={audit_item.get('reask')}；理由={audit_item.get('reask_reason') or '-'}"
        lines.extend([
            "",
            f"index={i}",
            f"当前speaker={s.get('speaker')} confidence={s.get('confidence')}",
            f"结构化依据: E={ev['E'] or '-'}; R={ev['R'] or '-'}; S={ev['S'] or '-'}",
            audit_line,
            f"候选/在场: {candidates or '（无）'}",
            f"前文: {str(s.get('context_before') or '')[-420:]}",
            f"台词: 「{s.get('text') or ''}」",
            f"后文: {str(s.get('context_after') or '')[:420]}",
        ])
    return "\n".join(lines)


def extract_json_array(text: str) -> list[dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def call_review(
    client: OpenAI,
    model: str,
    seg_name: str,
    parse_segments: list[dict[str, Any]],
    indices: list[int],
    timeout: int,
    audit: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    user = build_user_prompt(seg_name, parse_segments, indices, audit=audit)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=2500,
        timeout=timeout,
    )
    content = (resp.choices[0].message.content or "").strip()
    return extract_json_array(content)


def apply_reviews(
    parse_segments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    min_confidence: float,
    reason_gate: str = "none",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    updated = [dict(s) for s in parse_segments]
    stats = {
        "reviewed": 0,
        "revised": 0,
        "kept": 0,
        "uncertain": 0,
        "ignored": 0,
        "blocked_by_reason_gate": 0,
    }
    for r in reviews:
        try:
            i = int(r.get("index"))
        except (TypeError, ValueError):
            stats["ignored"] += 1
            continue
        if i < 0 or i >= len(updated):
            stats["ignored"] += 1
            continue
        decision = str(r.get("decision") or "").strip()
        speaker = str(r.get("speaker") or "").strip()
        try:
            conf = float(r.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        reason = str(r.get("reason") or "").strip()
        stats["reviewed"] += 1
        old = str(updated[i].get("speaker") or "").strip()
        if decision == "revise" and speaker == old:
            updated[i]["evidence"] = f"{updated[i].get('evidence') or ''}；证据复核确认({conf:.2f}): {reason}"
            stats["kept"] += 1
        elif decision == "revise" and speaker and conf >= min_confidence:
            if reason_gate == "strong" and not STRONG_REASON_RE.search(reason):
                updated[i]["evidence"] = f"{updated[i].get('evidence') or ''}；证据复核待人工({conf:.2f}): 反证不够强：{reason}"
                stats["blocked_by_reason_gate"] += 1
                stats["uncertain"] += 1
                continue
            old = updated[i].get("speaker")
            updated[i]["speaker"] = speaker
            updated[i]["confidence"] = max(conf, float(updated[i].get("confidence") or 0.0))
            updated[i]["evidence"] = f"证据复核改判：{old}→{speaker}({conf:.2f}); {reason}; {updated[i].get('evidence') or ''}"
            stats["revised"] += 1
        elif decision == "keep":
            updated[i]["evidence"] = f"{updated[i].get('evidence') or ''}；证据复核确认({conf:.2f}): {reason}"
            stats["kept"] += 1
        else:
            updated[i]["evidence"] = f"{updated[i].get('evidence') or ''}；证据复核待人工({conf:.2f}): {reason}"
            stats["uncertain"] += 1
    return updated, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", default="muli4_seg8")
    ap.add_argument("--parse", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/regression/evidence-review")
    ap.add_argument("--provider", default="tokenhub", choices=sorted(PROVIDERS))
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--target-mode", default="flagged",
                    choices=["flagged", "errors", "errors-and-controls", "audit-disagree", "audit-disagree-all"])
    ap.add_argument("--audit", type=Path, help="Optional machine-audit JSON with reask second opinions")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--min-confidence", type=float, default=0.85)
    ap.add_argument("--reason-gate", default="none", choices=["none", "strong"],
                    help="Automatic revise gate: strong requires explicit-counterevidence words in reviewer reason")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    key_name = {
        "agnes": "AGNES_API_KEY",
        "sensenova": "SENSENOVA_API_KEY",
        "tokenhub": "TOKENHUB_API_KEY",
    }[args.provider]
    api_key = os.environ.get(key_name)
    if not api_key:
        raise SystemExit(f"Missing {key_name}")

    parse_segments = load_segments(args.parse)
    audit = load_audit(args.audit)
    targets = choose_targets(args.seg, parse_segments, args.target_mode, args.limit, audit=audit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=PROVIDERS[args.provider], api_key=api_key)

    print(f"provider={args.provider} model={args.model} targets={len(targets)} mode={args.target_mode}")
    all_reviews: list[dict[str, Any]] = []
    for start in range(0, len(targets), args.batch_size):
        batch = targets[start:start + args.batch_size]
        print(f"review batch {start // args.batch_size + 1}: {batch}", flush=True)
        try:
            reviews = call_review(client, args.model, args.seg, parse_segments, batch, args.timeout, audit=audit)
        except Exception as exc:  # noqa: BLE001
            print(f"batch failed: {exc}", flush=True)
            reviews = []
        all_reviews.extend(reviews)
        time.sleep(0.5)

    updated, stats = apply_reviews(parse_segments, all_reviews, args.min_confidence, reason_gate=args.reason_gate)
    parse_out = args.out_dir / f"{args.seg}_parse.json"
    parse_out.write_text(json.dumps({"segments": updated, "stats": {"evidence_review": stats}},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    review_out = args.out_dir / f"{args.seg}_evidence_reviews.json"
    review_out.write_text(json.dumps({"targets": targets, "reviews": all_reviews, "stats": stats},
                                     ensure_ascii=False, indent=1), encoding="utf-8")

    score = score_sample(args.seg, parse_out)
    print(json.dumps({"stats": stats, "score": score}, ensure_ascii=False, indent=2))
    print(f"saved: {parse_out}")
    print(f"saved: {review_out}")


if __name__ == "__main__":
    main()
