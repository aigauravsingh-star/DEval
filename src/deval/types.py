from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isclose, isfinite
from typing import Any

from .jsonio import canonical, snapshot


KINDS = frozenset({"noul", "choice", "score"})


def number(value: Any, name: str, low: float = 0, high: float = 1) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}].")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return float(value)


def nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


@dataclass(frozen=True)
class Question:
    id: str
    instructions: str
    kind: str = "noul"
    criteria: Any = field(default_factory=dict)

    def __post_init__(self):
        nonempty(self.id, "Question id")
        nonempty(self.instructions, "Question instructions")
        if not isinstance(self.kind, str) or self.kind not in KINDS:
            raise ValueError("Question kind must be noul, choice, or score.")
        criteria = snapshot(self.criteria)
        if self.kind == "noul":
            if not isinstance(criteria, dict) or criteria and set(criteria) != {"true", "false"}:
                raise ValueError("Noul criteria must describe true and false, or be empty.")
        elif self.kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError("Choice needs at least two named options.")
            for key in criteria:
                nonempty(key, "Choice option")
        elif not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("Score needs at least two ordered levels, worst first.")
        values = criteria.values() if isinstance(criteria, dict) else criteria
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Question option descriptions must be non-empty strings.")
        object.__setattr__(self, "criteria", criteria)

    @property
    def options(self) -> tuple[str, ...]:
        if self.kind == "noul":
            return ("false", "true")
        return (
            tuple(self.criteria)
            if self.kind == "choice"
            else tuple(str(index) for index in range(len(self.criteria)))
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_payload(self) -> dict[str, Any]:
        result = {"type": self.kind, "instructions": self.instructions}
        if self.criteria:
            result["criteria"] = snapshot(self.criteria)
        return result


@dataclass(frozen=True)
class Answer:
    kind: str
    probabilities: dict[str, float]

    def __post_init__(self):
        if not isinstance(self.kind, str) or self.kind not in KINDS:
            raise ValueError("Unknown decision answer kind.")
        if not isinstance(self.probabilities, dict) or len(self.probabilities) < 2:
            raise ValueError("A decision answer requires a full probability distribution.")
        probabilities = {}
        for key, value in self.probabilities.items():
            nonempty(key, "Probability option")
            probabilities[key] = number(value, "Probability")
        if not isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one; no automatic renormalization.")
        object.__setattr__(self, "probabilities", probabilities)

    def validate(self, question: Question) -> None:
        if self.kind != question.kind or set(self.probabilities) != set(question.options):
            raise ValueError("Answer type/options do not match the requested question.")
        Answer(self.kind, self.probabilities)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    def __post_init__(self):
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token counts must be non-negative integers or unknown.")
        if self.cost_usd is not None:
            number(self.cost_usd, "Cost", high=float("inf"))


@dataclass(frozen=True)
class DecisionBatch:
    answers: dict[str, Answer]
    resolved_model: str | None = None
    usage: Usage = field(default_factory=Usage)

    def validate(self, questions: tuple[Question, ...]) -> None:
        if not isinstance(self.answers, dict) or set(self.answers) != {q.id for q in questions}:
            raise ValueError("A batch must answer exactly the requested question IDs.")
        if self.resolved_model is not None:
            nonempty(self.resolved_model, "Resolved model")
        if not isinstance(self.usage, Usage):
            raise ValueError("Backend usage must be Usage.")
        self.usage.__post_init__()
        for question in questions:
            answer = self.answers[question.id]
            if not isinstance(answer, Answer):
                raise ValueError("Backend answers must be typed Answer objects.")
            answer.validate(question)


@dataclass(frozen=True)
class Capabilities:
    kinds: frozenset[str] = KINDS
    native_probabilities: bool = True
    max_state_bytes: int = 1_048_576
    max_request_bytes: int = 2_097_152
    max_questions: int = 128
    max_choice_options: int = 255
    max_score_levels: int = 10

    def __post_init__(self):
        kinds = frozenset(self.kinds)
        if not kinds or not kinds <= KINDS:
            raise ValueError("Capabilities contain unsupported question kinds.")
        object.__setattr__(self, "kinds", kinds)
        if type(self.native_probabilities) is not bool:
            raise ValueError("native_probabilities must be a boolean.")
        for value in (
            self.max_state_bytes,
            self.max_request_bytes,
            self.max_questions,
            self.max_choice_options,
            self.max_score_levels,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("Capability limits must be positive integers.")

    def check(self, state: Any, questions: tuple[Question, ...]) -> None:
        if not self.native_probabilities:
            raise ValueError("Probability-based evaluation requires native probabilities.")
        if any(question.kind not in self.kinds for question in questions):
            raise ValueError("The backend does not support a requested question kind.")
        if len(questions) > self.max_questions:
            raise ValueError("The request exceeds the configured question limit.")
        if any(q.kind == "choice" and len(q.options) > self.max_choice_options for q in questions):
            raise ValueError("A question exceeds the configured choice-option limit.")
        if any(q.kind == "score" and len(q.options) > self.max_score_levels for q in questions):
            raise ValueError("A question exceeds the configured score-level limit.")
        if len(canonical(state).encode()) > self.max_state_bytes:
            raise ValueError(
                "The state exceeds the configured byte limit; it will not be truncated."
            )

    def to_dict(self):
        return {**asdict(self), "kinds": sorted(self.kinds), "modalities": ["text", "json"]}


@dataclass(frozen=True)
class EvalCase:
    id: str
    state: Any
    labels: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        nonempty(self.id, "Case id")
        if not isinstance(self.state, (str, dict, list)):
            raise ValueError("State must be text or a JSON object/array, not binary media.")
        object.__setattr__(self, "state", snapshot(self.state))
        if not isinstance(self.labels, dict):
            raise ValueError("Labels must be a question-ID mapping.")
        object.__setattr__(self, "labels", snapshot(self.labels))

    def to_dict(self):
        return asdict(self)
