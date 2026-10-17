# DEavl

**Provider-independent, decision-native evaluation for AI applications.**

Write bounded evaluation questions once. Use Jev, self-hosted Kev, Microsoft
Foundry, or another compatible System One server by changing configuration.
Keep full probability distributions and turn them into explicit rubric scores.
Run standalone, or use optional DeepEval and Ragas adapters.

DEavl is an independent Apache-2.0 alpha library. It has no runtime dependencies
in its core and does not require JevRails. It does not claim every model is
automatically compatible, equally calibrated, or equally accurate.

## Install and Try Without a Key

From this checkout, with Python 3.10+:

```bash
python -m pip install -e ".[dev]"
python examples/offline_eval.py
```

The demo uses fixed fixture answers. It is not inference and does not establish
judge quality, latency, or cost savings. There is no fallback from a live provider
to fixtures. This repository release is not a claim that the package is on PyPI.

## Run a Real Evaluation

Start your decision server or deploy a hosted model first. For a running local
Kev server:

```bash
deavl run examples/cases.jsonl --rubric examples/rubric.json --config examples/providers/kev.json --output kev-report.json --benchmark
```

For Jev, set `JEV_API_KEY` in your process environment and change only `--config`
to `examples/providers/jev.json`. Hosted providers receive the evaluation states
and questions. See [provider setup](docs/providers.md) before sending private data.

The example cases are two illustrative labels, not a security benchmark.
Exit `0` means all cases passed, `2` means a case failed/skipped, and `1` means a
configuration or execution error. The run JSON records scores, errors, and provenance;
`--benchmark` also creates an independent-label statistics file.

## Python API

```python
from deavl import EvalCase, Evaluator, Metric, Question, Rubric, load_backend

rubric = Rubric("support", version="1", metrics=(
    Metric(Question("grounded", "Is every fact in state.response supported by state.context?"),
           threshold=0.8, required=True),
    Metric(Question("relevant", "Does state.response answer state.input?")),
))
cases = [EvalCase("example", {
    "input": "When does the shop open?",
    "response": "At 9am.",
    "context": ["The shop opens at 9am."],
})]
report = Evaluator(load_backend({"provider": "kev"})).evaluate(cases, rubric)
print(report.summary)
```

Each case sends one request containing all rubric questions. The two concerns are
separate: the backend supplies semantic judgments; DEavl applies documented math.
`Evaluator.a_evaluate()` runs the synchronous workflow in a worker thread.

## What Is Included

- Typed Noul, Choice, and ordered Score questions, with validated distributions.
- Weighted scores, explicit option credits, higher/lower direction, hard requirements,
  and not-applicable handling.
- Configurable provider capabilities and byte/question limits; no silent truncation.
- Versioned rubrics, dataset/backend hashes, sent-state hashes, resolved model IDs,
  latency, token usage, and optional explicitly priced inference cost.
- Reference-label benchmarking: accuracy, confusion counts, false passes/failures,
  Brier score, clipped log loss, reliability bins/ECE, ordinal MAE, threshold sweeps,
  and binary abstention coverage.
- Saved-run comparison using identical data/rubrics/preparation and paired score deltas.
- Optional DeepEval metric and native System One model adapters; a Ragas numeric metric.
- JSONL datasets, JSON reports, CLI, and source/wheel packaging.

## Existing Frameworks

[DeepEval already supports JevEval](https://deepeval.com/docs/metrics-jev-eval) and
a custom System One model interface. [Ragas supports custom metrics](https://docs.ragas.io/en/stable/references/metrics/).
DEavl complements them with a small portable engine, configuration presets,
transparent scoring, and judge benchmarking. It does not claim to invent
decision-model evaluation or to reproduce all their built-in metrics.

```bash
python -m pip install -e ".[deepeval,ragas]"
```

See [framework integration examples](docs/integrations.md) and the
[scoring and benchmark definitions](docs/scoring.md).

## Boundaries

Only text/JSON states are supported in this release. Multimodal input, arbitrary
chat endpoints, and providers without native probabilities need other adapters
or are rejected. Capability declarations are configured assertions, not automatic
model certification. Do not relabel a chat model's self-reported certainty as a
calibrated probability.

Holistic groundedness is not Ragas claim-level faithfulness. Extraction needs
supplied claims or a separately versioned preparation function. No LLM is
implicitly called to extract claims or write explanations. The adapter's reasons
describe arithmetic outcomes, not model-generated evidence.

Reports omit inputs by default, but case IDs, rubric text, probabilities, endpoint
metadata, and hashes may still be sensitive. Custom backends own their identity
metadata. Core DEavl does not send telemetry or upload reports automatically;
optional frameworks have their own settings. See [security guidance](SECURITY.md).

The core test suite is offline, including loopback HTTP tests. Optional integration
tests exercise actual framework packages without model calls. Live provider access,
model accuracy, hardware support, and performance require separate validation.

