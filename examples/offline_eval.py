"""Recorded fixture scores only: no model calls, no quality or performance benchmark."""

import argparse
import json
from pathlib import Path

from deavl import Answer, Evaluator, FixtureBackend, Rubric, benchmark, read_jsonl
from deavl.jsonio import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="DEavl fixture demo, not model inference.")
    parser.add_argument("--output", help="Optional run JSON path; also writes benchmark statistics.")
    args = parser.parse_args()
    root = Path(__file__).parent
    rubric = Rubric.from_dict(read_json(root / "rubric.json"))
    cases = read_jsonl(root / "cases.jsonl")
    backend = FixtureBackend(
        {
            "grounded": Answer("noul", {"false": 0.1, "true": 0.9}),
            "quality": Answer("score", {"0": 0.1, "1": 0.2, "2": 0.7}),
            "handling": Answer(
                "choice", {"supported": 0.8, "fabricated": 0.1, "not_applicable": 0.1}
            ),
        }
    )
    report = Evaluator(backend).evaluate(cases, rubric)
    stats = benchmark(report, cases)
    if args.output:
        write_json(args.output, report.to_dict())
        write_json(args.output + ".benchmark.json", stats)
    print("DEavl fixture demo: fixed scores, not model inference.")
    print(json.dumps({"summary": report.summary, "judge_metrics": {
        key: {"accuracy": value["accuracy"], "brier_score": value["brier_score"],
              "ece": value["ece"], "false_passes": value["judge_pass_confusion"]["false_positive"]}
        for key, value in stats["metrics"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
