# Contributing to DEavl

Thanks for helping improve decision-model evaluation. Small, testable changes
and clear bug reports are especially useful while the library is in alpha.

## Set Up Locally

Requires Python 3.10+:

```bash
git clone https://github.com/aigauravsingh-star/DEval.git
cd DEval
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests examples
python examples/offline_eval.py
python -m build
```

The core tests and demo do not need model credentials. Optional native-framework
tests require the `deepeval` and `ragas` extras. CI exercises Python 3.10-3.12 and
the supported framework versions documented in [the integration guide](docs/integrations.md).

## Make a Useful Change

- Open an issue describing the problem, expected behavior, and a small reproduction.
- Keep pull requests focused and add regression tests for behavioral changes.
- For providers, test the typed request/response contract and error paths without
  requiring live credentials. Do not return fixture answers on inference failures.
- For scoring changes, document the math, missing-data behavior, and benchmark impact.
- Use synthetic or explicitly shareable test cases. Never include keys, private
  prompts, customer records, or proprietary reports.
- Label live-model results with the model revision, dataset, and methodology.
  Do not treat fixed fixture scores as model-quality evidence.

## Security Reports

Do not post exploitable vulnerabilities or secrets in a public issue. Follow
[SECURITY.md](SECURITY.md). Contributions are made under the repository's
[Apache 2.0 license](LICENSE).
