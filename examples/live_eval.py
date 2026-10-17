"""Calls the provider you configure. Hosted providers receive the selected case states."""

import argparse
import json
from pathlib import Path

from deval import Evaluator, Rubric, load_backend, read_jsonl
from deval.jsonio import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="report.json")
    args = parser.parse_args()
    root = Path(__file__).parent
    evaluator = Evaluator(load_backend(args.config))
    report = evaluator.evaluate(
        read_jsonl(root / "cases.jsonl"),
        Rubric.from_dict(read_json(root / "rubric.json")),
        on_error="record",
    )
    write_json(args.output, report.to_dict())
    print(json.dumps(report.summary, indent=2))
    return 1 if report.summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
