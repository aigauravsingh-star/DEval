from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass

from ..evaluator import Evaluator
from ..jsonio import snapshot
from ..types import EvalCase, Question


def _json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return snapshot(value)


def as_deepeval_metric(backend, rubric):
    """Wrap a DEavl rubric as a DeepEval BaseMetric; no generative judge involved."""
    try:
        from deepeval.metrics import BaseMetric
    except ImportError:
        raise ImportError("Install DEavl[deepeval] to use this integration.") from None

    class DecisionMetric(BaseMetric):
        def __init__(self):
            self.threshold = rubric.threshold
            self.evaluation_model = backend.identity().get("model") or "custom-decision-model"
            self.async_mode = True
            self.strict_mode = False
            self.include_reason = True
            self.score = self.success = self.reason = self.error = None
            self.evaluation_cost = None

        @property
        def __name__(self):
            return rubric.name

        def measure(self, test_case, *args, **kwargs):
            self.score = self.success = self.reason = self.error = None
            self.evaluation_cost = None
            try:
                state = {"input": test_case.input, "response": test_case.actual_output}
                if state["input"] is None or state["response"] is None:
                    raise ValueError("A single-turn test case needs input and actual_output.")
                for source, target in (
                    ("retrieval_context", "context"),
                    ("expected_output", "reference"),
                    ("tools_called", "tool_calls"),
                ):
                    value = getattr(test_case, source, None)
                    if value is not None:
                        state[target] = _json_value(value)
                tools = state.get("tool_calls", [])
                state["tool_results"] = [item["output"] for item in tools if "output" in item]
                report = Evaluator(backend).evaluate([EvalCase("deepeval", state)], rubric)
                row = report.results[0]
                if row.score is None:
                    raise ValueError("No applicable metric decisions.")
                self.score, self.success = row.score, row.passed
                self.evaluation_cost = row.cost_usd
                self.reason = "; ".join(
                    f"{key}: value={value['value']}, passed={value['passed']}"
                    for key, value in row.metrics.items()
                )
                return self.score
            except Exception as exc:
                self.error = f"Decision evaluation failed ({type(exc).__name__})."
                self.success = False
                raise RuntimeError(self.error) from None

        async def a_measure(self, test_case, *args, **kwargs):
            return await asyncio.to_thread(self.measure, test_case, *args, **kwargs)

        def is_successful(self):
            return self.error is None and self.success is True

    return DecisionMetric()


def as_deepeval_model(backend):
    """Expose a backend through DeepEval's native System One model interface."""
    try:
        from deepeval.models import DeepEvalBaseSystemOneModel
        from deepeval.models.system_one.schema import (
            ChoiceAnswer,
            ChoiceQuestion,
            NoulAnswer,
            NoulQuestion,
            ScoreAnswer,
            ScoreQuestion,
            SystemOneAnswers,
        )
    except ImportError:
        raise ImportError("Install DEavl[deepeval] with the System One model interface.") from None

    class DecisionModel(DeepEvalBaseSystemOneModel):
        def __init__(self):
            self.backend = backend
            super().__init__(model=backend.identity().get("model") or "custom-decision-model")

        def load_model(self):
            return self.backend

        def get_model_name(self):
            return self.name

        def decide(self, state, questions):
            converted = []
            for key, q in questions.items():
                if isinstance(q, NoulQuestion):
                    criteria = (
                        {}
                        if q.true is None and q.false is None
                        else {
                            "true": str(q.true)
                            if q.true is not None
                            else "The proposition is true.",
                            "false": str(q.false)
                            if q.false is not None
                            else "The proposition is false.",
                        }
                    )
                    converted.append(Question(key, q.instructions, "noul", criteria))
                elif isinstance(q, ChoiceQuestion):
                    converted.append(
                        Question(
                            key,
                            q.instructions,
                            "choice",
                            {
                                option: str(description) if description is not None else option
                                for option, description in q.options.items()
                            },
                        )
                    )
                elif isinstance(q, ScoreQuestion):
                    converted.append(Question(key, q.instructions, "score", list(q.levels)))
                else:
                    raise ValueError("Unsupported DeepEval decision question type.")
            converted = tuple(converted)
            backend.capabilities.check(state, converted)
            batch = backend.decide(state, converted)
            batch.validate(converted)
            result = SystemOneAnswers()
            for q in converted:
                answer = batch.answers[q.id]
                probs = answer.probabilities
                # This is distribution concentration, not a new probability of correctness.
                confidence = (len(probs) * max(probs.values()) - 1) / (len(probs) - 1)
                confidence = min(1.0, max(0.0, confidence))
                if q.kind == "noul":
                    result.nouls[q.id] = NoulAnswer(probability=probs["true"])
                elif q.kind == "choice":
                    result.choices[q.id] = ChoiceAnswer(
                        choice=max(q.options, key=probs.__getitem__),
                        probabilities=probs,
                        confidence=confidence,
                    )
                else:
                    result.scores[q.id] = ScoreAnswer(
                        score=sum(int(k) * p for k, p in probs.items()),
                        probabilities={int(k): p for k, p in probs.items()},
                        confidence=confidence,
                    )
            return result, batch.usage.cost_usd

        async def a_decide(self, state, questions):
            return await asyncio.to_thread(self.decide, state, questions)

    return DecisionModel()
