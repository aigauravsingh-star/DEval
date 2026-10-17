import asyncio
from dataclasses import replace

import pytest

from deval import (
    Answer,
    Capabilities,
    DecisionBatch,
    EvalCase,
    Evaluator,
    FixtureBackend,
    Metric,
    Question,
    Rubric,
    RunReport,
    Usage,
    benchmark,
    compare_runs,
)
from deval.jsonio import fingerprint, loads


def make_rubric():
    return Rubric("test", "1", (Metric(Question("correct", "Is this correct?")),))


def run(p=0.9, cases=None, rubric=None):
    backend = FixtureBackend({"correct": Answer("noul", {"false": 1 - p, "true": p})})
    return Evaluator(backend).evaluate(
        cases or [EvalCase("1", "text", {"correct": True})], rubric or make_rubric()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": ""},
        {"instructions": ""},
        {"kind": "chat"},
        {"criteria": {"true": "yes"}},
        {"criteria": {"true": "", "false": "no"}},
        {"kind": "choice", "criteria": {"one": "Only option"}},
        {"kind": "score", "criteria": ["one"]},
    ],
)
def test_question_validation(kwargs):
    with pytest.raises(ValueError):
        Question(**{"id": "id", "instructions": "question", **kwargs})


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {"a": 1},
        {"a": 0.8, "b": 0.8},
        {"a": -0.1, "b": 1.1},
        {"a": True, "b": False},
        {"a": "0.5", "b": "0.5"},
        {"a": float("nan"), "b": 0.5},
        {"a": float("inf"), "b": 0.5},
    ],
)
def test_distribution_validation(probabilities):
    with pytest.raises(ValueError):
        Answer("choice", probabilities)


def test_distribution_must_match_question():
    with pytest.raises(ValueError):
        Answer("choice", {"a": 0.5, "b": 0.5}).validate(Question("id", "test"))


def test_scoring_semantics():
    q = Question("risk", "Is this risky?")
    assert (
        Metric(q, direction="lower").score(Answer("noul", {"false": 0.9, "true": 0.1}))["value"]
        == 0.9
    )
    q = Question("grade", "Quality?", "score", ["poor", "fair", "good"])
    assert Metric(q).score(Answer("score", {"0": 0.1, "1": 0.2, "2": 0.7}))[
        "value"
    ] == pytest.approx(0.8)
    q = Question("label", "Behavior?", "choice", {"a": "Good", "b": "Bad", "na": "Not applicable"})
    metric = Metric(q, credits={"a": 1, "b": 0, "na": None})
    assert metric.score(Answer("choice", {"a": 0.6, "b": 0.2, "na": 0.2}))[
        "value"
    ] == pytest.approx(0.75)
    assert metric.score(Answer("choice", {"a": 0.1, "b": 0.1, "na": 0.8}))["value"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": 0},
        {"weight": True},
        {"threshold": -0.1},
        {"required": "false"},
        {"direction": "sideways"},
        {"credits": {"true": 1}},
        {"not_applicable_threshold": 0},
    ],
)
def test_metric_validation(kwargs):
    with pytest.raises(ValueError):
        Metric(Question("id", "question"), **kwargs)


def test_choice_requires_explicit_credit():
    with pytest.raises(ValueError):
        Metric(Question("id", "Which?", "choice", {"a": "A", "b": "B"}))


def test_required_failure_cannot_be_hidden_by_weighted_average():
    rubric = Rubric(
        "hard",
        "1",
        (
            Metric(Question("required", "Safe?"), weight=1, required=True),
            Metric(Question("quality", "Good?"), weight=100),
        ),
    )
    backend = FixtureBackend(
        {
            "required": Answer("noul", {"false": 0.9, "true": 0.1}),
            "quality": Answer("noul", {"false": 0, "true": 1}),
        }
    )
    row = Evaluator(backend).evaluate([EvalCase("1", "text")], rubric).results[0]
    assert row.score > 0.9
    assert row.passed is False


def test_report_has_provenance_without_input_or_credentials():
    report = run(cases=[EvalCase("1", "PRIVATE_TEXT", {"correct": True})])
    data = report.to_dict()
    assert "PRIVATE_TEXT" not in str(data)
    assert data["manifest"]["rubric_hash"] == fingerprint(make_rubric().to_dict())
    assert data["manifest"]["dataset_hash"]
    assert data["summary"]["cost_usd"] is None
    assert data["summary"]["unknown_cost_cases"] == 1
    assert RunReport.from_dict(data).summary == report.summary


def test_errors_are_not_scores_and_exception_bodies_are_private():
    class Broken(FixtureBackend):
        def decide(self, state, questions):
            raise RuntimeError("PRIVATE_TEXT and SECRET_TOKEN")

    evaluator = Evaluator(Broken({}))
    with pytest.raises(RuntimeError) as exc:
        evaluator.evaluate([EvalCase("1", "text")], make_rubric())
    assert "PRIVATE_TEXT" not in str(exc.value)
    report = evaluator.evaluate([EvalCase("1", "text")], make_rubric(), on_error="record")
    assert report.results[0].score is None
    assert report.results[0].error == "RuntimeError"
    assert report.summary["mean_score"] is None
    assert report.summary["pass_rate"] is None
    assert report.summary["errors"] == 1
    assert "SECRET_TOKEN" not in str(report.to_dict())


def test_bad_and_missing_backend_answers_are_errors():
    class Missing(FixtureBackend):
        def decide(self, state, questions):
            return DecisionBatch({})

    report = Evaluator(Missing({})).evaluate(
        [EvalCase("1", "text")], make_rubric(), on_error="record"
    )
    assert report.results[0].status == "error"


def test_skipped_is_not_passed():
    question = Question("correct", "Which?", "choice", {"a": "Good", "na": "N/A"})
    rubric = Rubric("na", "1", (Metric(question, credits={"a": 1, "na": None}, required=True),))
    backend = FixtureBackend({"correct": Answer("choice", {"a": 0, "na": 1})})
    report = Evaluator(backend).evaluate([EvalCase("1", "text")], rubric)
    assert report.summary["skipped"] == 1
    assert report.results[0].passed is None
    assert report.results[0].score is None


def test_prepare_hook_version_and_sent_state_hash():
    with pytest.raises(ValueError):
        Evaluator(FixtureBackend({}), prepare=lambda case: case.state)
    backend = FixtureBackend({"correct": Answer("noul", {"false": 0.1, "true": 0.9})})
    result = Evaluator(
        backend, prepare=lambda c: {"safe": "masked"}, prepare_version="mask-1"
    ).evaluate([EvalCase("1", "private")], make_rubric())
    assert result.manifest["prepare_version"] == "mask-1"
    assert result.results[0].sent_state_hash == fingerprint({"safe": "masked"})


def test_unsupported_capability_is_rejected_before_inference():
    class Narrow(FixtureBackend):
        capabilities = Capabilities(kinds=frozenset({"choice"}))

        def decide(self, *args):
            raise AssertionError("Must not call inference")

    row = (
        Evaluator(Narrow({}))
        .evaluate([EvalCase("1", "text")], make_rubric(), on_error="record")
        .results[0]
    )
    assert row.error == "ValueError"


@pytest.mark.parametrize(
    "cap",
    [
        {"native_probabilities": False},
        {"max_state_bytes": 1},
        {"max_questions": 1},
    ],
)
def test_capabilities(cap):
    capabilities = Capabilities(**cap)
    questions = (Question("a", "A?"), Question("b", "B?"))
    with pytest.raises(ValueError):
        capabilities.check("too long", questions)


def test_async_evaluation():
    backend = FixtureBackend({"correct": Answer("noul", {"false": 0.1, "true": 0.9})})
    report = asyncio.run(Evaluator(backend).a_evaluate([EvalCase("1", "text")], make_rubric()))
    assert report.results[0].passed


def test_dataset_ids_and_json_are_strict():
    with pytest.raises(ValueError):
        run(cases=[EvalCase("same", "a"), EvalCase("same", "b")])
    with pytest.raises(ValueError):
        EvalCase("1", {"bad": float("nan")})
    with pytest.raises(ValueError):
        loads('{"x":1,"x":2}')


def test_benchmark_known_binary_statistics_and_thresholds():
    cases = [EvalCase("1", "text", {"correct": True}), EvalCase("2", "text", {"correct": False})]
    report = run(0.8, cases)
    item = benchmark(report, cases)["metrics"]["correct"]
    assert item["accuracy"] == 0.5
    assert item["brier_score"] == pytest.approx(0.34)
    assert item["ece"] == pytest.approx(0.3)
    assert item["confusion"]["false_positive"] == 1
    assert item["judge_pass_confusion"]["false_positive"] == 1
    assert item["selective"]["coverage_all_cases"] == 1
    assert item["threshold_sweep"][-1]["false_negative"] == 1


def test_risk_direction_changes_false_pass_interpretation():
    rubric = Rubric("risk", "1", (Metric(Question("correct", "Risk present?"), direction="lower"),))
    cases = [EvalCase("1", "text", {"correct": True})]
    report = run(0.1, cases, rubric)
    item = benchmark(report, cases)["metrics"]["correct"]
    assert item["confusion"]["false_negative"] == 1
    assert item["judge_pass_confusion"]["false_positive"] == 1


def test_multiclass_and_ordinal_benchmarks():
    question = Question("grade", "Grade?", "score", ["bad", "fair", "good"])
    rubric = Rubric("grades", "1", (Metric(question),))
    cases = [EvalCase("1", "text", {"grade": 2})]
    backend = FixtureBackend({"grade": Answer("score", {"0": 0.1, "1": 0.2, "2": 0.7})})
    item = benchmark(Evaluator(backend).evaluate(cases, rubric), cases)["metrics"]["grade"]
    assert item["accuracy"] == 1
    assert item["expected_level_mae"] == pytest.approx(0.4)
    assert item["brier_score"] == pytest.approx(0.14)


def test_benchmark_rejects_missing_gold_or_changed_dataset():
    cases = [EvalCase("1", "text")]
    report = run(cases=cases)
    with pytest.raises(ValueError, match="Missing independent"):
        benchmark(report, cases)
    with pytest.raises(ValueError, match="hash"):
        benchmark(report, [EvalCase("1", "changed", {"correct": True})])


def test_all_errors_remain_visible_in_benchmark():
    class Broken(FixtureBackend):
        def decide(self, *args):
            raise RuntimeError("offline")

    cases = [EvalCase("1", "text", {"correct": True})]
    report = Evaluator(Broken({})).evaluate(cases, make_rubric(), on_error="record")
    item = benchmark(report, cases)["metrics"]["correct"]
    assert item["accuracy"] is None
    assert item["scored_cases"] == 0
    assert item["error_cases"] == 1


def test_comparison_requires_identical_inputs_and_uses_paired_cases():
    cases = [EvalCase("1", "text", {"correct": True}), EvalCase("2", "text", {"correct": True})]
    left, right = run(0.8, cases), run(0.9, cases)
    right = replace(right, results=tuple(reversed(right.results)))
    result = compare_runs([left, right], cases)
    assert result["pairwise_application_score_deltas"][0]["paired_cases"] == 2
    assert result["pairwise_application_score_deltas"][0]["mean_right_minus_left"] == pytest.approx(
        0.1
    )
    changed = run(cases=[EvalCase("1", "different", {"correct": True})])
    with pytest.raises(ValueError):
        compare_runs([left, changed], cases)


def test_report_tampering_is_rejected():
    data = run().to_dict()
    data["manifest"]["rubric"]["version"] = "tampered"
    with pytest.raises(ValueError):
        RunReport.from_dict(data)


def test_cost_unknown_is_not_zero():
    class Costed(FixtureBackend):
        def decide(self, state, questions):
            return DecisionBatch(self.answers, "pinned", Usage(10, 0, 0.2))

    report = Evaluator(Costed({"correct": Answer("noul", {"false": 0.1, "true": 0.9})})).evaluate(
        [EvalCase("1", "text")], make_rubric()
    )
    assert report.summary["cost_usd"] == 0.2


def test_reference_labels_are_never_sent_to_backend():
    seen = []

    class Capture(FixtureBackend):
        def decide(self, state, questions):
            seen.append(state)
            return super().decide(state, questions)

    cases = [EvalCase("1", {"input": "public"}, {"correct": True})]
    Evaluator(Capture({"correct": Answer("noul", {"false": 0.1, "true": 0.9})})).evaluate(
        cases, make_rubric()
    )
    assert seen == [{"input": "public"}]


def test_mutating_original_rubric_cannot_change_inflight_scoring():
    question = Question("correct", "Which?", "choice", {"a": "good", "b": "bad"})
    credits = {"a": 1, "b": 0}
    rubric = Rubric("snapshot", "1", (Metric(question, credits=credits),))

    class Mutate(FixtureBackend):
        def decide(self, state, questions):
            rubric.metrics[0].credits["a"] = 0
            return super().decide(state, questions)

    report = Evaluator(Mutate({"correct": Answer("choice", {"a": 0.9, "b": 0.1})})).evaluate(
        [EvalCase("1", "text")], rubric
    )
    assert report.results[0].score == 0.9
    assert report.manifest["rubric"]["metrics"][0]["credits"]["a"] == 1


@pytest.mark.parametrize("kind,label", [("noul", 1), ("score", True), ("choice", "unknown")])
def test_reference_label_types_are_not_coerced(kind, label):
    q = (
        Question("correct", "True?")
        if kind == "noul"
        else Question(
            "correct", "Which?", kind, ["bad", "good"] if kind == "score" else {"a": "A", "b": "B"}
        )
    )
    metric = Metric(q, credits={"a": 1, "b": 0} if kind == "choice" else None)
    cases = [EvalCase("1", "text", {"correct": label})]
    distribution = {key: 1 / len(q.options) for key in q.options}
    report = Evaluator(FixtureBackend({"correct": Answer(kind, distribution)})).evaluate(
        cases, Rubric("test", "1", (metric,))
    )
    with pytest.raises(ValueError):
        benchmark(report, cases)


def test_inclusive_threshold_is_stable_under_arithmetic_rounding():
    q = Question("grade", "Grade?", "score", ["bad", "fair", "good"])
    assert (
        Metric(q, threshold=0.8).score(Answer("score", {"0": 0.1, "1": 0.2, "2": 0.7}))["passed"]
        is True
    )
    assert (
        Metric(q, threshold=0.81).score(Answer("score", {"0": 0.1, "1": 0.2, "2": 0.7}))["passed"]
        is False
    )


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_option_limits_are_enforced_before_inference(kind):
    question = Question(
        "id",
        "Which?",
        kind,
        {"a": "A", "b": "B", "c": "C"} if kind == "choice" else ["bad", "fair", "good"],
    )
    with pytest.raises(ValueError):
        Capabilities(max_choice_options=2, max_score_levels=2).check("state", (question,))
