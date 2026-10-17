import asyncio
import socket

import pytest

from deavl import Answer, FixtureBackend, Metric, Question, Rubric


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    monkeypatch.setenv("CONFIDENT_TELEMETRY_OPT_OUT", "YES")
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")

    def forbidden(*args, **kwargs):
        raise AssertionError("Integration fixture tests must not open network connections.")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def backend_and_rubric():
    backend = FixtureBackend({"relevant": Answer("noul", {"false": 0.1, "true": 0.9})})
    rubric = Rubric(
        "relevance",
        "1",
        (Metric(Question("relevant", "Does state.response address state.input?")),),
    )
    return backend, rubric


def test_real_deepeval_metric_and_async():
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase
    from deavl.integrations.deepeval import as_deepeval_metric

    backend, rubric = backend_and_rubric()
    metric = as_deepeval_metric(backend, rubric)
    case = LLMTestCase(input="Question", actual_output="Answer")
    assert metric.measure(case) == pytest.approx(0.9)
    assert metric.is_successful()
    assert metric.evaluation_cost is None
    assert asyncio.run(metric.a_measure(case)) == pytest.approx(0.9)
    backend.answers.clear()
    with pytest.raises(RuntimeError):
        metric.measure(case)
    assert metric.score is None
    assert not metric.is_successful()


def test_real_deepeval_system_one_interface_all_types():
    pytest.importorskip("deepeval")
    from deepeval.models.system_one.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
    from deavl.integrations.deepeval import as_deepeval_model

    backend = FixtureBackend(
        {
            "yes": Answer("noul", {"false": 0.1, "true": 0.9}),
            "label": Answer("choice", {"a": 0.9, "b": 0.1}),
            "grade": Answer("score", {"0": 0.1, "1": 0.9}),
        }
    )
    model = as_deepeval_model(backend)
    questions = {
        "yes": NoulQuestion(instructions="True?"),
        "label": ChoiceQuestion(instructions="Which?", options={"a": "A", "b": "B"}),
        "grade": ScoreQuestion(instructions="Grade?", levels=["bad", "good"]),
    }
    answers, cost = model.decide("state", questions)
    assert answers.nouls["yes"].probability == 0.9
    assert answers.choices["label"].choice == "a"
    assert answers.scores["grade"].normalized == 0.9
    assert cost is None
    async_answers, _ = asyncio.run(model.a_decide("state", questions))
    assert async_answers.nouls["yes"].probability == 0.9


def test_real_jev_eval_accepts_deavl_model():
    pytest.importorskip("deepeval")
    from deepeval.metrics import JevEval
    from deepeval.metrics.jev_eval import Noul
    from deepeval.test_case import LLMTestCase, SingleTurnParams
    from deavl import DecisionBatch, DecisionBackend
    from deavl.integrations.deepeval import as_deepeval_model

    class AlwaysYes(DecisionBackend):
        def identity(self):
            return {"provider": "fixture", "model": "fixture"}

        def decide(self, state, questions):
            return DecisionBatch(
                {q.id: Answer("noul", {"false": 0.1, "true": 0.9}) for q in questions}
            )

    metric = JevEval(
        name="portable",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        questions=[Noul("The output answers the input.")],
        system_one_model=as_deepeval_model(AlwaysYes()),
        async_mode=False,
        include_reason=False,
    )
    metric.measure(LLMTestCase(input="Question", actual_output="Answer"))
    assert metric.score == pytest.approx(0.9)


def test_real_ragas_numeric_metric():
    pytest.importorskip("ragas")
    from deavl.integrations.ragas import as_ragas_metric

    backend, rubric = backend_and_rubric()
    metric = as_ragas_metric(backend, rubric)
    result = metric.score(user_input="Question", response="Answer", retrieved_contexts=["Evidence"])
    assert result.value == pytest.approx(0.9)
