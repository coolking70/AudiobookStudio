"""Export the local silver correction store for offline training/evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning_store import read_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + ("\n" if records else ""), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
