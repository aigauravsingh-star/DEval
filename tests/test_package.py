from importlib.metadata import distribution
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import deval
from deval import (
    Answer,
    EvalCase,
    Evaluator,
    FixtureBackend,
    Metric,
    Question,
    Rubric,
    RunReport,
    SystemOneBackend,
)
from deval.cli import main


def test_distribution_and_cli_use_deval():
    package = distribution("DEval")
    assert package.metadata["Name"] == "DEval"
    assert package.version == deval.__version__ == "0.2.0"
    commands = {entry.name: entry.value for entry in package.entry_points}
    assert commands["deval"] == "deval.cli:main"
    assert "deavl" not in commands
    assert Path(deval.__file__).parent.name == "deval"


def test_cli_help_uses_renamed_command(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.startswith("usage: deval ")


def test_reports_record_current_library_version():
    backend = FixtureBackend({"ok": Answer("noul", {"false": 0.1, "true": 0.9})})
    rubric = Rubric("test", "1", (Metric(Question("ok", "Is the response correct?")),))
    report = Evaluator(backend).evaluate([EvalCase("one", "answer")], rubric)
    assert report.manifest["library_version"] == deval.__version__
    old_report = report.to_dict()
    old_report["manifest"]["library_version"] = "0.1.0"
    assert RunReport.from_dict(old_report).manifest["library_version"] == "0.1.0"


def test_user_agent_matches_name_and_version():
    backend = SystemOneBackend(base_url="http://localhost:8009", model="test")
    assert backend._headers()["User-Agent"] == f"DEval/{deval.__version__}"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_branding_wordmark_uses_deval(theme):
    root = Path(__file__).resolve().parents[1]
    tree = ET.parse(root / "docs" / "assets" / f"hero-{theme}.svg")
    ns = "{http://www.w3.org/2000/svg}"
    assert tree.find(ns + "title").text == "DEval"
    wordmark = tree.find(ns + "text")
    assert "".join(wordmark.itertext()) == "DEval"
