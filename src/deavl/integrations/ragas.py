from __future__ import annotations

from ..evaluator import Evaluator
from ..types import EvalCase


def as_ragas_metric(backend, rubric):
    """A native Ragas numeric metric for custom scoring/experiments, not a replacement LLM."""
    try:
        from ragas.metrics import numeric_metric
    except ImportError:
        raise ImportError("Install DEavl[ragas] to use this integration.") from None

    @numeric_metric(name=rubric.name, allowed_values=(0.0, 1.0))
    def decision_score(
        user_input: str,
        response: str,
        retrieved_contexts: list[str] | None = None,
        reference: str | None = None,
    ) -> float:
        state = {"input": user_input, "response": response}
        if retrieved_contexts is not None:
            state["context"] = retrieved_contexts
        if reference is not None:
            state["reference"] = reference
        row = Evaluator(backend).evaluate([EvalCase("ragas", state)], rubric).results[0]
        if row.score is None:
            raise ValueError("No applicable metric decisions; this is not a zero score.")
        return row.score

    return decision_score
