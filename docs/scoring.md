# Scoring and Benchmark Definitions

## Application Evaluation

Questions address only the state you supply. DEavl does not retrieve evidence or
invent missing context. A prompt-injection-resistant judge still requires testing;
untrusted evaluation content can attack the evaluator itself.

`Question(id, instructions)` is Noul by default. Choice takes a dictionary of
named descriptions; Score takes an ordered list of levels, worst first.
Each backend must return a complete typed probability distribution. Values must
be finite and in [0, 1], all option keys must match, and probabilities must sum to
one within 1e-5 rounding tolerance. Raw distributions are retained, not silently
rewritten. Labels alone or a scalar confidence are insufficient.

For Noul the default credits are false=0, true=1. A yes probability is the chance
the proposition is true, not the chance the answer is useful or the evaluator is
correct. For Score the default credit of level k is k/(number of levels - 1).
Choice metrics require explicit credit for every option; credit is a number in
[0,1], or null for an inapplicable option. Noul/Score can also override credits.

The metric value is sum(probability * credit) / applicable probability mass.
If excluded mass is at least `not_applicable_threshold` (default 0.5), the metric
is skipped. Zero applicable mass is skipped too. The denominator means even tiny
provider rounding deviations do not create out-of-range values. With
`direction="lower"`, value becomes 1-value; use this for risk questions where
yes means a violation. Labels and raw probabilities are never inverted.

A metric passes when its value reaches its threshold, with a 1e-12 absolute
tolerance for floating-point rounding. An application score
is the weighted mean over applicable metrics. A case passes only if that score
reaches the rubric threshold **and every required metric passes**. Thus a weighted
average cannot hide a failed hard requirement. A skipped required metric prevents
passing. If no metrics apply, the whole case is skipped, not passed or assigned zero.

This is transparent aggregation, not a proof that the aggregate is itself a
calibrated probability. Fixed arithmetic does not guarantee repeatable model outputs.

## Preparation and Extraction

The optional preparation hook can mask data, select fields, or attach claims
extracted by a separate system. It requires a version for run provenance:

```python
from deavl import EvalCase, Evaluator

def prepare(case: EvalCase):
    # supplied_claims must come from your extraction process, not this evaluator.
    return {"response": case.state["response"],
            "context": case.state["context"],
            "claims": case.state["supplied_claims"]}

evaluator = Evaluator(backend, prepare=prepare, prepare_version="claim-schema-1")
```

Record extraction-model/config changes in that version. Build explicit claim-level
questions when you need a claim-level denominator; a single holistic groundedness
question is a different metric from Ragas faithfulness. The hook's latency is
included, but its costs are not inferred. Do not leak reference labels into judge
state. Cases keep `labels` separate and the built-in evaluator never sends them.

## Judge Benchmarking

Supply independent human/adjudicated labels per question in `EvalCase.labels`:

- Noul: a boolean describing whether the proposition is true.
- Choice: the reference option key.
- Score: an integer level index, starting at zero.

The reference dataset must match the run's dataset hash, including its labels.
Use held-out data; do not treat a model judging itself as independent ground truth.
An error row is excluded from accuracy denominators but explicitly counted.
All-error runs report null accuracy and calibration, not perfect or zero accuracy.

Metrics include:

- Accuracy: Noul uses P(true) >= the benchmark threshold (default 0.5); other
  types use the highest-probability option; ties use canonical key order for
  Choice and declared level order for Score.
- Brier score: mean (P(true)-label)^2 for Noul; mean sum over options of
  (probability-one_hot_label)^2 for Choice/Score. The definitions differ by a
  factor of two for equivalent binary categorical distributions, so compare like types.
- Log loss: mean -log(reference probability), clipped at 1e-15 with a clipping count.
- ECE: fixed equal-width bins. Noul compares P(true) with true-label frequency;
  other types compare top-option probability with top-option accuracy. Both
  output bin counts/means. ECE depends on binning and sample size.
- Ordinal MAE: distance between expected level index and the reference level.
- Binary confusion and threshold sweeps: positive means the proposition is true.
  For a risk question, false negative is a missed risk, not a false rejection.
- Judge-pass confusion: compare the scored decision's pass status with the
  pass status obtained from the reference's one-hot distribution using the same
  direction, credits, and threshold. False positive here is a false pass, false
  negative a false fail. Inapplicable labels/decisions are excluded and counted.
- Binary selective coverage: retain only probabilities strictly outside the
  abstention band (default inclusive [0.3, 0.7]); coverage uses all reference
  cases, including errors, and accuracy uses retained cases only.

These are analyses, not automatic recalibration. Select thresholds on development
data and report final performance on separate test data. No universal safe
threshold or accuracy/latency/cost improvement is promised.

## Comparing Runs

`compare_runs` requires identical dataset hashes, rubric hashes, preparation
versions, case IDs, and sent-state hashes. Per-run statistics retain their error
counts. Paired application-score deltas use cases scored successfully by every
run and report the paired count. These deltas describe judges' scores; they do
not establish which judge is more correct. Use the independent-label statistics
for that comparison. No statistical significance is inferred from a small delta.

Reports record timestamp/run ID, package/schema version, public backend identity,
declared capabilities, requested/resolved models, rubric definition/version,
dataset hash, per-case sent-state hash, decisions, latency, and known/unknown cost.
Aliases and a server-reported model string do not prove a checkpoint is pinned.
Hashes identify artifacts, but are not signatures, encryption, or authentication.
Store immutable datasets/configs/model revisions alongside your reports.

`on_error="raise"` is the default. `on_error="record"` preserves error rows with
null score/pass status and only the exception class; no exception message, prompt,
or provider response body is copied into those rows. Custom identity metadata and
rubrics can still contain sensitive content. Reports are data, not public logs.
