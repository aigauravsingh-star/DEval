from __future__ import annotations

import argparse
import json
import sys

from .benchmark import benchmark, compare_runs
from .config import load_backend
from .datasets import read_jsonl
from .evaluator import Evaluator, RunReport
from .jsonio import read_json, write_json
from .rubrics import Rubric


def main(argv=None):
    parser = argparse.ArgumentParser(prog="deavl")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Evaluate JSONL cases with a configured provider.")
    run.add_argument("dataset")
    run.add_argument("--config", required=True)
    run.add_argument("--rubric", required=True)
    run.add_argument("--output", required=True)
    run.add_argument(
        "--benchmark",
        action="store_true",
        help="Require independent labels and add judge statistics.",
    )
    bench = commands.add_parser(
        "benchmark", help="Benchmark a saved run against independent labels."
    )
    bench.add_argument("report")
    bench.add_argument("dataset")
    bench.add_argument("--output", required=True)
    compare = commands.add_parser(
        "compare", help="Compare saved runs on the same labelled dataset."
    )
    compare.add_argument("dataset")
    compare.add_argument("reports", nargs="+")
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            cases = read_jsonl(args.dataset)
            rubric = Rubric.from_dict(read_json(args.rubric))
            evaluator = Evaluator(load_backend(args.config))
            if args.benchmark:
                # Validate reference labels before incurring inference cost.
                from .benchmark import _label

                for case in cases:
                    for metric in rubric.metrics:
                        _label(metric.question, case.labels[metric.question.id])
            report = evaluator.evaluate(cases, rubric, on_error="record")
            write_json(args.output, report.to_dict())
            if args.benchmark:
                write_json(args.output + ".benchmark.json", benchmark(report, cases))
            print(json.dumps(report.summary, allow_nan=False))
            return (
                1
                if report.summary["errors"]
                else 2
                if any(row.passed is False or row.status == "skipped" for row in report.results)
                else 0
            )
        cases = read_jsonl(args.dataset)
        if args.command == "benchmark":
            result = benchmark(RunReport.from_dict(read_json(args.report)), cases)
        else:
            result = compare_runs(
                [RunReport.from_dict(read_json(path)) for path in args.reports], cases
            )
        write_json(args.output, result)
        print(json.dumps({"output": args.output}))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        # Provider exception bodies can include prompts; the report carries error classes only.
        print(
            f"DEavl: command failed ({type(exc).__name__}). Check config, schemas, and permissions.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
