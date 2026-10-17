# Provider Setup

`load_backend` accepts a trusted JSON path, dictionary, `BackendConfig`, or no
argument to read `DEAVL_CONFIG` / `DEAVL_*` environment settings. A supplied
file/object wins over environment configuration. A `DEAVL_CONFIG` file wins over
individual field variables. Secrets are resolved only from the named environment
variable. Files never contain literal credentials, and `.env` is not auto-loaded.

## Presets

| Provider | Defaults | Authentication |
| --- | --- | --- |
| `jev` | TypeSafe URL, `/v1/systemone`, stable `jev-1.13.0` | Bearer; `JEV_API_KEY`, then `TYPESAFE_API_KEY`, unless an explicit variable is named |
| `kev` | `http://127.0.0.1:8009`, `/v1/systemone`, `kev-latest` | None unless explicitly configured |
| `azure` | Microsoft provider path; URL/deployment required | `api-key` by default, or bearer/Entra |
| `systemone` | `/v1/systemone`; URL/model required | None unless explicitly configured |

Only Jev uses the TypeSafe credential defaults. No cross-provider credential or
inference fallback occurs. Example files are in `examples/providers/`.

For a compatible Laya, H2O-Lightning, Decision 3.0, or other server, use `systemone`
with that server's URL, model ID, and supported `kinds`. Wire compatibility does
not establish evaluation quality. Servers with different APIs need a Python
`DecisionBackend` implementation. Generic endpoints are not chat-completion endpoints.

## Kev

Follow the [upstream installation guide](https://github.com/jaredpalmer/kev).
The server has separate Python/runtime requirements; do not install the unrelated
PyPI package named `kev`. For an existing source checkout:

```bash
uv sync --locked --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --host 127.0.0.1 --port 8009
```

Initial model downloads require network/storage. `kev-latest` in requests addresses
the server's loaded checkpoint; pin the serving weights and code separately.
CPU/smaller models have different latency and accuracy. Keep state truncation
disabled. If a response reports truncation, DEavl rejects it.

For authenticated Kev, set `KEV_API_KEY` on the server. Set the matching secret
in the client process and add `"auth": "bearer", "api_key_env": "KEV_API_KEY"`.
An existing key is not sent unless explicitly referenced. Keep local unauthenticated
servers bound to loopback. Use HTTPS and access controls for remote deployment.

## Azure

Use your Foundry resource URL and **deployment name**, not necessarily the model's
catalog ID. The endpoint defaults to `/providers/microsoft/v1/systemone`.
See [Microsoft's guide](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision).

`auth: "api_key"` reads `AZURE_API_KEY`; `auth: "bearer"` reads `AZURE_ENTRA_TOKEN`
unless `api_key_env` is explicit. Static credentials are read once at construction.
For automatically renewed Entra tokens:

```bash
python -m pip install -e ".[azure]"
```

Use `auth: "entra"` with no `api_key_env`. Azure's `DefaultAzureCredential` is
queried on each request and handles its own caching/renewal. Identity and RBAC
are configured outside DEavl. Foundry still processes evaluation data externally.

## Configuration Fields

Each field has an environment equivalent named `DEAVL_` plus its uppercase name:

- `provider`, `model`, `base_url`, `endpoint` select the backend and route.
- `auth`: `none`, `bearer`, `api_key`, or Azure-only `entra`.
- `api_key_env`: credential variable name, never the secret value.
- `timeout`: 0.001 through 3600 seconds, default 30.
- `allow_insecure_http`: boolean, default false; only loopback allows plain HTTP by default.
- `kinds`: subset of `noul`, `choice`, `score`; comma-separated in the environment.
- `max_state_bytes`: default 1 MiB; `max_request_bytes`: default 2 MiB.
- `max_questions`: default 128. Byte limits are not tokenizer/context-window guarantees.
- `max_choice_options`: default 255; `max_score_levels`: default 10. Set these
  to the limits supported by your actual server/model.
- `input_usd_per_million`, `output_usd_per_million`: optional explicit non-negative prices.

Both prices and both token counts must be known for computed cost. Missing cost
stays null, including self-hosted compute; no automatic price lookup or claim of
free inference is made. Prices are captured in the run identity. Preparation,
main-application inference, infrastructure, and failed-request charges are not inferred.

Unknown fields, missing credentials, invalid limits, query/userinfo URLs, malformed
answers, and incompatible capabilities are rejected. No automatic retry or
redirect is followed. Backend error bodies are not included in report errors.
Model servers can still omit truncation markers or misreport capabilities; the
library cannot certify an arbitrary server's internal behavior.
