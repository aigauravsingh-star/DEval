from __future__ import annotations

from math import log
from statistics import mean
from typing import Iterable

from .evaluator import RunReport, validate_report
from .jsonio import fingerprint
from .types import Answer, EvalCase, number


def _label(question, value):
    if question.kind == "noul":
        if type(value) is not bool:
            raise ValueError("Noul reference labels must be booleans.")
        return "true" if value else "false"
    if question.kind == "score":
        if type(value) is not int or not 0 <= value < len(question.options):
            raise ValueError("Score labels must be valid integer level indices.")
        return str(value)
    if not isinstance(value, str) or value not in question.options:
        raise ValueError("Choice labels must be declared option keys.")
    return value


def _bins(pairs, count):
    bins = []
    for i in range(count):
        group = [(p, y) for p, y in pairs if min(int(p * count), count - 1) == i]
        bins.append(
            {
                "lower": i / count,
                "upper": (i + 1) / count,
                "count": len(group),
                "mean_probability": mean(p for p, _ in group) if group else None,
                "observed_frequency": mean(y for _, y in group) if group else None,
            }
        )
    total = len(pairs)
    ece = (
        sum(
            b["count"] * abs(b["mean_probability"] - b["observed_frequency"])
            for b in bins
            if b["count"]
        )
        / total
        if total
        else None
    )
    return bins, ece


def _binary(predictions):
    tp = sum(pred and gold for pred, gold in predictions)
    fp = sum(pred and not gold for pred, gold in predictions)
    fn = sum(not pred and gold for pred, gold in predictions)
    tn = sum(not pred and not gold for pred, gold in predictions)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
    }


def benchmark(
    report: RunReport,
    cases: Iterable[EvalCase],
    *,
    bins: int = 10,
    threshold: float = 0.5,
    abstain_band: tuple[float, float] = (0.3, 0.7),
    thresholds: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> dict:
    """Measure decisions against independent labels, not another model's opinion."""
    rubric = validate_report(report)
    cases = tuple(cases)
    if fingerprint([c.to_dict() for c in cases]) != report.manifest["dataset_hash"]:
        raise ValueError("Benchmark dataset/labels do not match the evaluated dataset hash.")
    if type(bins) is not int or not 1 <= bins <= 1000:
        raise ValueError("bins must be an integer from 1 to 1000.")
    number(threshold, "Binary threshold")
    low, high = abstain_band
    number(low, "Abstention lower bound")
    number(high, "Abstention upper bound")
    if low >= high:
        raise ValueError("Abstention lower bound must be less than upper bound.")
    for t in thresholds:
        number(t, "Sweep threshold")
    by_id = {c.id: c for c in cases}
    if set(by_id) != {r.case_id for r in report.results}:
        raise ValueError("Report case IDs do not match the reference dataset.")
    results = {}
    for metric in rubric.metrics:
        q = metric.question
        labels = {}
        for case in cases:
            if q.id not in case.labels:
                raise ValueError(f"Missing independent reference label for '{q.id}'.")
            labels[case.id] = _label(q, case.labels[q.id])
        points = []
        for row in report.results:
            if row.status == "error":
                continue
            raw = row.metrics.get(q.id)
            if not isinstance(raw, dict):
                raise ValueError("Report is missing a metric decision.")
            answer = Answer(q.kind, raw["probabilities"])
            answer.validate(q)
            probs = answer.probabilities
            predicted = (
                ("true" if probs["true"] >= threshold else "false")
                if q.kind == "noul"
                else max(q.options, key=probs.__getitem__)
            )
            points.append((probs, labels[row.case_id], predicted))
        n = len(points)
        calibration_pairs = [
            (p["true"], float(y == "true"))
            if q.kind == "noul"
            else (max(p.values()), float(y == prediction))
            for p, y, prediction in points
        ]
        reliability, ece = _bins(calibration_pairs, bins)
        briers = [
            (p["true"] - (y == "true")) ** 2
            if q.kind == "noul"
            else sum((p[k] - (k == y)) ** 2 for k in q.options)
            for p, y, _ in points
        ]
        item = {
            "labelled_cases": len(cases),
            "scored_cases": n,
            "error_cases": len(cases) - n,
            "accuracy": mean(pred == y for _, y, pred in points) if n else None,
            "brier_score": mean(briers) if n else None,
            "brier_definition": "binary P(true) squared error"
            if q.kind == "noul"
            else "sum of multiclass squared errors",
            "log_loss": mean(-log(max(p[y], 1e-15)) for p, y, _ in points) if n else None,
            "log_loss_clip_epsilon": 1e-15,
            "clipped_probabilities": sum(p[y] < 1e-15 for p, y, _ in points),
            "ece": ece,
            "reliability": reliability,
            "calibration_target": "P(true) vs true frequency"
            if q.kind == "noul"
            else "top-option confidence vs accuracy",
        }
        pass_pairs = []
        for p, gold, _ in points:
            actual_pass = metric.score(Answer(q.kind, p))["passed"]
            ideal_pass = metric.score(Answer(q.kind, {k: float(k == gold) for k in q.options}))[
                "passed"
            ]
            if actual_pass is not None and ideal_pass is not None:
                pass_pairs.append((actual_pass, ideal_pass))
        item["judge_pass_confusion"] = {"applicable_cases": len(pass_pairs), **_binary(pass_pairs)}
        if q.kind == "score":
            item["expected_level_mae"] = (
                mean(abs(sum(int(k) * v for k, v in p.items()) - int(y)) for p, y, _ in points)
                if n
                else None
            )
        if q.kind == "noul":
            item["confusion"] = _binary([(pred == "true", y == "true") for _, y, pred in points])
            retained = [
                (p["true"] >= threshold, y == "true")
                for p, y, _ in points
                if not low <= p["true"] <= high
            ]
            item["selective"] = {
                "abstain_band": [low, high],
                "retained": len(retained),
                "coverage_all_cases": len(retained) / len(cases),
                "accuracy_retained": mean(pred == y for pred, y in retained) if retained else None,
            }
            item["threshold_sweep"] = [
                {"threshold": t, **_binary([(p["true"] >= t, y == "true") for p, y, _ in points])}
                for t in thresholds
            ]
        results[q.id] = item
    return {
        "run_id": report.manifest["run_id"],
        "dataset_hash": report.manifest["dataset_hash"],
        "rubric_hash": report.manifest["rubric_hash"],
        "summary": report.summary,
        "metrics": results,
    }


def compare_runs(reports: Iterable[RunReport], cases: Iterable[EvalCase]) -> dict:
    reports, cases = tuple(reports), tuple(cases)
    if len(reports) < 2:
        raise ValueError("Comparison requires at least two runs.")
    for report in reports:
        validate_report(report)
    signatures = {
        (r.manifest["dataset_hash"], r.manifest["rubric_hash"], r.manifest["prepare_version"])
        for r in reports
    }
    if len(signatures) != 1:
        raise ValueError("Compare identical datasets, rubrics, and preparation versions.")
    sent = [{row.case_id: row.sent_state_hash for row in run.results} for run in reports]
    if any(value != sent[0] for value in sent[1:]):
        raise ValueError("Prepared/sent states differ across runs.")
    comparable = set.intersection(
        *[{row.case_id for row in run.results if row.status != "error"} for run in reports]
    )
    base_rows = {row.case_id: row for row in reports[0].results}
    deltas = []
    for other in reports[1:]:
        other_rows = {row.case_id: row for row in other.results}
        pairs = [
            (base_rows[key].score, other_rows[key].score)
            for key in sorted(comparable)
            if base_rows[key].score is not None and other_rows[key].score is not None
        ]
        deltas.append(
            {
                "left_run": reports[0].manifest["run_id"],
                "right_run": other.manifest["run_id"],
                "paired_cases": len(pairs),
                "mean_right_minus_left": mean(b - a for a, b in pairs) if pairs else None,
            }
        )
    return {
        "runs": [
            {"backend": run.manifest["backend"], "benchmark": benchmark(run, cases)}
            for run in reports
        ],
        "common_success_cases": len(comparable),
        "pairwise_application_score_deltas": deltas,
        "warning": "Per-run metrics may have different error exclusions; paired deltas use common scored cases. Model-judged application scores are not independent gold labels.",
    }
