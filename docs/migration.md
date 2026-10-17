# Migrating to DEval 0.2

The project is now named **DEval**, matching its GitHub repository. Version 0.2.0
renames the Python package, CLI, and configuration prefix. Evaluation algorithms,
provider presets, and report schema version 1 are unchanged.

## Update an Existing Checkout

Inside your checkout, after updating its source:

```bash
python -m pip uninstall DEavl
python -m pip install -e .
deval --help
```

Install from this repository, not an assumed PyPI release. If you used an editable
install and renamed your checkout directory, reinstall it from the new location.

## Update Imports and Commands

| Before | Now |
| --- | --- |
| Distribution name `DEavl` | `DEval` |
| `from deavl import Evaluator` | `from deval import Evaluator` |
| `deavl.integrations.deepeval` | `deval.integrations.deepeval` |
| `deavl run ...` | `deval run ...` |
| `DEAVL_CONFIG` | `DEVAL_CONFIG` |
| `DEAVL_PROVIDER` and other `DEAVL_*` settings | `DEVAL_PROVIDER` and `DEVAL_*` |

The old import, CLI, and environment prefix are not aliases. Update deployment
settings and scripts together. Provider credential variables such as `JEV_API_KEY`,
`TYPESAFE_API_KEY`, and `AZURE_API_KEY` keep their existing names. Explicit
`api_key_env` values are user-selected names and are not rewritten automatically.

Provider JSON fields, rubric JSON, datasets, and existing schema-version-1 reports
do not need renaming. New reports record library version 0.2.0. The GitHub URL
remains https://github.com/aigauravsingh-star/DEval.
