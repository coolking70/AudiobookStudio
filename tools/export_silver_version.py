"""Export quality-gated silver corrections as an immutable dataset bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import learning_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-records", type=int, default=20)
    args = parser.parse_args()
    try:
        bundle = learning_store.build_version_bundle(min_records=args.min_records)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "dataset_id": bundle["dataset_id"],
        "records": bundle["record_count"],
        "sha256": bundle["sha256"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
