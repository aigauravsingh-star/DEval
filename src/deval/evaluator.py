from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from typing import Any
from uuid import uuid4

from .backends import DecisionBackend
from .jsonio import canonical, fingerprint, snapshot
from .rubrics import Rubric, reaches
from .types import DecisionBatch, EvalCase, number


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    status: str
    score: float | None
    passed: bool | None
    metrics: dict[str, dict[str, Any]]
    latency_ms: float
    cost_usd: float | None
    resolved_model: str | None
    sent_state_hash: str
    error: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RunReport:
    manifest: dict[str, Any]
    results: tuple[EvalResult, ...]

    @property
    def summary(self):
        ok = [r for r in self.results if r.status == "ok"]
        latencies = sorted(r.latency_ms for r in self.results)
        costs = [r.cost_usd for r in self.results]
        return {
            "cases": len(self.results),
            "scored": len(ok),
            "errors": sum(r.status == "error" for r in self.results),
            "skipped": sum(r.status == "skipped" for r in self.results),
            "mean_score": mean(r.score for r in ok) if ok else None,
            "pass_rate": mean(r.passed for r in ok) if ok else None,
            "latency_p50_ms": _percentile(latencies, 0.5),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
            "known_cost_subtotal_usd": sum(c for c in costs if c is not None),
            "unknown_cost_cases": sum(c is None for c in costs),
        }

    def to_dict(self):
        return {
            "manifest": snapshot(self.manifest),
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"manifest", "summary", "results"}:
            raise ValueError("Invalid run report shape.")
        try:
            results = tuple(EvalResult(**row) for row in data["results"])
            report = cls(snapshot(data["manifest"]), results)
            validate_report(report)
        except (TypeError, KeyError):
            raise ValueError("Invalid run report fields.") from None
        return report


def _percentile(values, q):
    if not values:
        return None
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def validate_report(report: RunReport):
    if report.manifest.get("schema_version") != "1":
        raise ValueError("Unsupported report schema version.")
    rubric = Rubric.from_dict(report.manifest["rubric"])
    if fingerprint(rubric.to_dict()) != report.manifest["rubric_hash"]:
        raise ValueError("Report rubric hash does not match its contents.")
    if fingerprint(report.manifest["backend"]) != report.manifest["backend_hash"]:
        raise ValueError("Report backend hash does not match its contents.")
    if len({r.case_id for r in report.results}) != len(report.results):
        raise ValueError("Duplicate report case IDs.")
    for result in report.results:
        number(result.latency_ms, "Reported latency", high=float("inf"))
        if result.cost_usd is not None:
            number(result.cost_usd, "Reported cost", high=float("inf"))
        if result.status not in {"ok", "error", "skipped"}:
            raise ValueError("Unknown report result status.")
        if result.status == "error":
            if result.score is not None or result.passed is not None or not result.error:
                raise ValueError("Errors must not contain evaluation scores.")
        elif result.status == "ok":
            number(result.score, "Reported score")
            if type(result.passed) is not bool:
                raise ValueError("Scored results require a boolean verdict.")
        elif result.score is not None or result.passed is not None:
            raise ValueError("Skipped results must not contain evaluation scores.")
    return rubric


class Evaluator:
    def __init__(
        self,
        backend: DecisionBackend,
        *,
        prepare: Callable[[EvalCase], Any] | None = None,
        prepare_version: str | None = None,
    ):
        if not isinstance(backend, DecisionBackend):
            raise ValueError("Provide a DecisionBackend adapter.")
        if prepare is not None and (not callable(prepare) or not prepare_version):
            raise ValueError("A preparation/extraction hook needs an explicit prepare_version.")
        self.backend, self.prepare, self.prepare_version = backend, prepare, prepare_version

    def evaluate(
        self, cases: Iterable[EvalCase], rubric: Rubric, *, on_error: str = "raise"
    ) -> RunReport:
        if on_error not in {"raise", "record"}:
            raise ValueError("on_error must be raise or record.")
        rubric = Rubric.from_dict(deepcopy(rubric.to_dict()))
        cases = tuple(EvalCase(**deepcopy(case.to_dict())) for case in cases)
        if not cases or len({c.id for c in cases}) != len(cases):
            raise ValueError("Supply a non-empty dataset with unique case IDs.")
        identity = snapshot(self.backend.identity())
        questions = tuple(m.question for m in rubric.metrics)
        manifest = {
            "schema_version": "1",
            "library_version": "0.2.0",
            "run_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_hash": fingerprint([case.to_dict() for case in cases]),
            "rubric": rubric.to_dict(),
            "rubric_hash": fingerprint(rubric.to_dict()),
            "backend": identity,
            "backend_hash": fingerprint(identity),
            "capabilities": self.backend.capabilities.to_dict(),
            "prepare_version": self.prepare_version,
            "cost_scope": "decision_backend_only; extraction/preparation costs are not inferred",
        }
        rows = []
        for case in cases:
            started = perf_counter()
            state_hash = fingerprint(case.state)
            try:
                state = snapshot(self.prepare(deepcopy(case)) if self.prepare else case.state)
                if not isinstance(state, (str, dict, list)):
                    raise ValueError("Prepared state must be text or a JSON object/array.")
                state_hash = fingerprint(state)
                self.backend.capabilities.check(state, questions)
                batch = self.backend.decide(state, deepcopy(questions))
                if not isinstance(batch, DecisionBatch):
                    raise ValueError("Backend must return DecisionBatch.")
                batch.validate(questions)
                metrics = {
                    m.question.id: m.score(batch.answers[m.question.id]) for m in rubric.metrics
                }
                applicable = [
                    m for m in rubric.metrics if metrics[m.question.id]["value"] is not None
                ]
                score = (
                    sum(metrics[m.question.id]["value"] * m.weight for m in applicable)
                    / sum(m.weight for m in applicable)
                    if applicable
                    else None
                )
                passed = (
                    None
                    if score is None
                    else reaches(score, rubric.threshold)
                    and all(
                        metrics[m.question.id]["passed"] is True
                        for m in rubric.metrics
                        if m.required
                    )
                )
                rows.append(
                    EvalResult(
                        case.id,
                        "ok" if score is not None else "skipped",
                        score,
                        passed,
                        metrics,
                        (perf_counter() - started) * 1000,
                        batch.usage.cost_usd,
                        batch.resolved_model,
                        state_hash,
                    )
                )
            except Exception as exc:
                if on_error == "raise":
                    raise RuntimeError(
                        f"Evaluation failed for case '{case.id}' ({type(exc).__name__})."
                    ) from None
                rows.append(
                    EvalResult(
                        case.id,
                        "error",
                        None,
                        None,
                        {},
                        (perf_counter() - started) * 1000,
                        None,
                        None,
                        state_hash,
                        type(exc).__name__,
                    )
                )
        # Detect accidental policy mutation during a run instead of silently changing its definition.
        if canonical(rubric.to_dict()) != canonical(manifest["rubric"]):
            raise RuntimeError("The rubric changed during evaluation.")
        return RunReport(manifest, tuple(rows))

    async def a_evaluate(self, cases, rubric, *, on_error="raise"):
        return await asyncio.to_thread(self.evaluate, cases, rubric, on_error=on_error)
