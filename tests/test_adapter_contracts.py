"""Small protocol stubs test our wrappers; separate tests exercise the real frameworks."""

import asyncio
import sys
import types

import pytest

from deavl import Answer, FixtureBackend, Metric, Question, Rubric


def setup_metric():
    backend = FixtureBackend({"correct": Answer("noul", {"false": 0.1, "true": 0.9})})
    rubric = Rubric("test", "1", (Metric(Question("correct", "Correct?")),))
    return backend, rubric


def test_deepeval_metric_contract_and_error_reset(monkeypatch):
    package = types.ModuleType("deepeval")
    module = types.ModuleType("deepeval.metrics")
    module.BaseMetric = type("BaseMetric", (), {})
    monkeypatch.setitem(sys.modules, "deepeval", package)
    monkeypatch.setitem(sys.modules, "deepeval.metrics", module)
    from deavl.integrations.deepeval import as_deepeval_metric

    backend, rubric = setup_metric()
    metric = as_deepeval_metric(backend, rubric)
    case = types.SimpleNamespace(input="Question", actual_output="Answer")
    assert metric.measure(case) == 0.9
    assert metric.is_successful()
    assert metric.evaluation_cost is None
    assert asyncio.run(metric.a_measure(case)) == 0.9
    backend.answers.clear()
    with pytest.raises(RuntimeError):
        metric.measure(case)
    assert metric.score is None
    assert not metric.is_successful()


def test_ragas_metric_contract(monkeypatch):
    module = types.ModuleType("ragas.metrics")

    def numeric_metric(**params):
        assert params["name"] == "test"
        assert params["allowed_values"] == (0, 1)

        def decorator(function):
            return types.SimpleNamespace(
                score=lambda **kwargs: types.SimpleNamespace(value=function(**kwargs))
            )

        return decorator

    module.numeric_metric = numeric_metric
    monkeypatch.setitem(sys.modules, "ragas", types.ModuleType("ragas"))
    monkeypatch.setitem(sys.modules, "ragas.metrics", module)
    from deavl.integrations.ragas import as_ragas_metric

    backend, rubric = setup_metric()
    metric = as_ragas_metric(backend, rubric)
    assert (
        metric.score(
            user_input="Question", response="Answer", retrieved_contexts=["Evidence"]
        ).value
        == 0.9
    )


def test_native_deepeval_model_contract(monkeypatch):
    class Base:
        def __init__(self, model):
            self.name = model
            self.model = self.load_model()

    class Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class NoulQuestion(Schema):
        true = false = None

    class ChoiceQuestion(Schema):
        pass

    class ScoreQuestion(Schema):
        pass

    class Answers:
        def __init__(self):
            self.nouls, self.choices, self.scores = {}, {}, {}

    models = types.ModuleType("deepeval.models")
    models.DeepEvalBaseSystemOneModel = Base
    schema = types.ModuleType("deepeval.models.system_one.schema")
    for key, value in {
        "NoulQuestion": NoulQuestion,
        "ChoiceQuestion": ChoiceQuestion,
        "ScoreQuestion": ScoreQuestion,
        "NoulAnswer": Schema,
        "ChoiceAnswer": Schema,
        "ScoreAnswer": Schema,
        "SystemOneAnswers": Answers,
    }.items():
        setattr(schema, key, value)
    for name, module in {
        "deepeval": types.ModuleType("deepeval"),
        "deepeval.models": models,
        "deepeval.models.system_one": types.ModuleType("deepeval.models.system_one"),
        "deepeval.models.system_one.schema": schema,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    from deavl.integrations.deepeval import as_deepeval_model

    backend = FixtureBackend(
        {
            "yes": Answer("noul", {"false": 0.1, "true": 0.9}),
            "choice": Answer("choice", {"a": 0.9, "b": 0.1}),
            "score": Answer("score", {"0": 0.1, "1": 0.9}),
        }
    )
    model = as_deepeval_model(backend)
    questions = {
        "yes": NoulQuestion(instructions="True?"),
        "choice": ChoiceQuestion(instructions="Which?", options={"a": "A", "b": "B"}),
        "score": ScoreQuestion(instructions="Grade?", levels=["bad", "good"]),
    }
    answers, cost = model.decide("text", questions)
    assert answers.nouls["yes"].probability == 0.9
    assert answers.choices["choice"].choice == "a"
    assert answers.scores["score"].score == 0.9
    assert cost is None
    result, _ = asyncio.run(model.a_decide("text", questions))
    assert result.nouls["yes"].probability == 0.9
