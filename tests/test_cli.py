import json

import pytest

from deval import Answer, FixtureBackend, Metric, Question, Rubric
from deval.cli import main
from deval.datasets import read_jsonl
from deval.jsonio import read_json, write_json


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    rubric = Rubric("test", "1", (Metric(Question("correct", "Correct?")),))
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"id":"1","state":{"response":"answer"},"labels":{"correct":true}}\n', encoding="utf-8"
    )
    rubric_path, config_path = tmp_path / "rubric.json", tmp_path / "backend.json"
    write_json(rubric_path, rubric.to_dict())
    write_json(config_path, {"provider": "kev"})
    backend = FixtureBackend({"correct": Answer("noul", {"false": 0.1, "true": 0.9})})
    monkeypatch.setattr("deval.cli.load_backend", lambda path: backend)
    return dataset, rubric_path, config_path, tmp_path / "report.json", backend


def test_cli_full_roundtrip_and_comparison(inputs, tmp_path):
    dataset, rubric, config, output, _ = inputs
    assert (
        main(
            [
                "run",
                str(dataset),
                "--rubric",
                str(rubric),
                "--config",
                str(config),
                "--output",
                str(output),
                "--benchmark",
            ]
        )
        == 0
    )
    assert read_json(str(output) + ".benchmark.json")["metrics"]["correct"]["accuracy"] == 1
    stats = tmp_path / "stats.json"
    assert main(["benchmark", str(output), str(dataset), "--output", str(stats)]) == 0
    assert main(["compare", str(dataset), str(output), str(output), "--output", str(stats)]) == 0
    assert read_json(stats)["common_success_cases"] == 1


def test_cli_metric_failure_has_separate_exit_code(inputs):
    dataset, rubric, config, output, backend = inputs
    backend.answers["correct"] = Answer("noul", {"false": 0.9, "true": 0.1})
    assert (
        main(
            [
                "run",
                str(dataset),
                "--rubric",
                str(rubric),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 2
    )


def test_cli_errors_saved_but_not_turned_into_passes(inputs):
    dataset, rubric, config, output, backend = inputs
    backend.answers.clear()
    assert (
        main(
            [
                "run",
                str(dataset),
                "--rubric",
                str(rubric),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert read_json(output)["results"][0]["score"] is None


def test_cli_reference_validation_happens_before_inference(inputs, monkeypatch):
    dataset, rubric, config, output, backend = inputs
    dataset.write_text('{"id":"1","state":"text"}', encoding="utf-8")

    def forbidden(*args):
        raise AssertionError("Must not incur inference cost")

    monkeypatch.setattr(backend, "decide", forbidden)
    assert (
        main(
            [
                "run",
                str(dataset),
                "--rubric",
                str(rubric),
                "--config",
                str(config),
                "--output",
                str(output),
                "--benchmark",
            ]
        )
        == 1
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "line",
    [
        "[]",
        '{"id":"1","state":"x","extra":1}',
        '{"id":"1","state":null}',
        '{"id":"1","state":{"x":NaN}}',
        '{"id":"1","state":"a","state":"b"}',
    ],
)
def test_dataset_validation(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(path)


def test_rubric_json_roundtrip(inputs):
    _, rubric, _, _, _ = inputs
    original = read_json(rubric)
    assert Rubric.from_dict(json.loads(json.dumps(original))).to_dict() == original
