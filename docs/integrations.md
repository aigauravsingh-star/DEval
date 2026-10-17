# Framework Integrations

Install the extras only for frameworks you use. The core does not import either
framework. Integration contracts target DeepEval 4.3.1 and Ragas 0.4.3; CI tests
these versions without any hosted model call. Pin your versions and re-run tests
when upgrading. Each framework has separate telemetry, logging, caching, and
upload settings; review those before processing sensitive data.

## DeepEval Metric

```python
from deepeval.test_case import LLMTestCase
from deavl import Metric, Question, Rubric, load_backend
from deavl.integrations.deepeval import as_deepeval_metric

rubric = Rubric("relevance", "1", (
    Metric(Question("relevant", "Does state.response answer state.input?")),
))
metric = as_deepeval_metric(load_backend({"provider": "kev"}), rubric)
metric.measure(LLMTestCase(input="When do you open?", actual_output="At 9am."))
print(metric.score, metric.is_successful())
```

This is a `BaseMetric` with synchronous and asynchronous measurement. Use it in
DeepEval's runner as you would a custom metric. It maps input/actual_output to
`state.input`/`state.response`, retrieval_context to context, expected_output to
reference, and tools_called to tool_calls plus extracted tool_results. Other
fields are not invented. Reasons summarize arithmetic outcomes, not a generated
rationale. Use `is_successful()` for required gates, not just score >= threshold.
Each metric instance is mutable, following DeepEval's model; do not concurrently
reuse one instance yourself. The built-in wrapper is for single-turn test cases.

## DeepEval Native System One Model

```python
from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Noul
from deepeval.test_case import SingleTurnParams
from deavl import load_backend
from deavl.integrations.deepeval import as_deepeval_model

model = as_deepeval_model(load_backend({"provider": "kev"}))
metric = JevEval(
    name="relevance",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    questions=[Noul("The actual_output answers the input.")],
    system_one_model=model,
)
```

The framework owns the questions, state names, aggregation, and runner in this
mode; DEavl supplies provider transport and typed decisions. Noul, Choice, and
Score are mapped explicitly. Structured/non-string question descriptions that
DEavl cannot represent are rejected. The adapter's confidence is distribution
concentration required by the framework, not extra evidence of correctness.

Only use the model for DeepEval modes that support its System One interface.
`GEval` or a generative extraction stage cannot use a decision model as a text
generator. Built-in metric availability and algorithms are framework-specific.

## Ragas Numeric Metric

```python
from deavl import Metric, Question, Rubric, load_backend
from deavl.integrations.ragas import as_ragas_metric

rubric = Rubric("relevance", "1", (
    Metric(Question("relevant", "Does state.response answer state.input?")),
))
metric = as_ragas_metric(load_backend({"provider": "kev"}), rubric)
result = metric.score(
    user_input="When do you open?",
    response="At 9am.",
    retrieved_contexts=["We open at 9am."],
)
print(result.value)
```

This wraps Ragas's native `numeric_metric` for custom scoring/experiments, not
the legacy `ragas.evaluate` collection-metric interface. It maps user_input to
input, response to response, retrieved_contexts to context, and reference to
reference. Ragas numeric values carry the weighted score only; hard requirements
and DEavl's overall pass verdict are not encoded in that scalar. For release
gates, use the standalone DEavl report or DeepEval metric adapter.

This is not a custom text-generation LLM wrapper and does not automatically
replace Ragas's built-in Faithfulness, embeddings, or synthetic-data generation.
Whole-case inapplicability and backend failures raise errors, never zero scores.

## Custom Backends

Subclass `DecisionBackend`, implement `decide(state, questions) -> DecisionBatch`
and `identity()`, and declare actual `Capabilities`. Return an `Answer` with the
full normalized option distribution per question, optional `Usage`, and a resolved
model revision when known. Do not put API keys in identity metadata. Inherit the
worker-thread async bridge or override `a_decide` for direct native async callers.
`Evaluator.a_evaluate` deliberately offloads the complete synchronous workflow;
cancelling the await does not terminate in-flight HTTP calls. Request timeouts still apply.
