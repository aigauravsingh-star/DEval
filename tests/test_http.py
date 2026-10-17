import io
import json
import os
import sys
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError

import pytest

from deavl import BackendConfig, Question, SystemOneBackend, load_backend


QUESTIONS = (
    Question("yes", "True?"),
    Question("label", "Which?", "choice", {"a": "A", "b": "B"}),
    Question("grade", "Grade?", "score", ["bad", "good"]),
)


def response():
    return {
        "model": "pinned-model",
        "answers": {
            "yes": {"type": "noul", "noul": 0.9},
            "label": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}},
            "grade": {"type": "score", "score": 0.9, "probabilities": {"0": 0.1, "1": 0.9}},
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("DEAVL_", "JEV_", "TYPESAFE_", "AZURE_", "KEV_")):
            monkeypatch.delenv(key, raising=False)


def test_parse_all_kinds_and_explicit_pricing():
    backend = SystemOneBackend(
        base_url="http://localhost:8009",
        model="m",
        input_usd_per_million=1,
        output_usd_per_million=0,
    )
    batch = backend.parse(response(), QUESTIONS)
    assert batch.answers["yes"].probabilities["true"] == 0.9
    assert batch.resolved_model == "pinned-model"
    assert batch.usage.cost_usd == pytest.approx(0.0001)
    assert (
        SystemOneBackend(base_url="http://localhost:8009", model="m")
        .parse(response(), QUESTIONS)
        .usage.cost_usd
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "wrong-type",
        "bad-probabilities",
        "confidence-only",
        "choice-conflict",
        "score-conflict",
        "truncated",
        "token-truncation",
        "bad-usage",
        "nan",
        "bool",
    ],
)
def test_parse_rejects_invalid_responses(change):
    data = response()
    if change == "missing":
        del data["answers"]["yes"]
    elif change == "extra":
        data["answers"]["extra"] = {"type": "noul", "noul": 0.1}
    elif change == "wrong-type":
        data["answers"]["yes"]["type"] = "score"
    elif change == "bad-probabilities":
        data["answers"]["label"]["probabilities"]["a"] = 0.2
    elif change == "confidence-only":
        data["answers"]["yes"] = {"type": "noul", "confidence": 0.9}
    elif change == "choice-conflict":
        data["answers"]["label"]["choice"] = "b"
    elif change == "score-conflict":
        data["answers"]["grade"]["score"] = 0.1
    elif change == "truncated":
        data["truncated"] = True
    elif change == "token-truncation":
        data["usage"].update(state_tokens=10, state_tokens_used=5)
    elif change == "bad-usage":
        data["usage"]["input_tokens"] = "100"
    else:
        data["answers"]["yes"]["noul"] = float("nan") if change == "nan" else True
    with pytest.raises(ValueError):
        SystemOneBackend(base_url="http://localhost:8009", model="m").parse(data, QUESTIONS)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://remote.test",
        "https://user:key@example.test",
        "https://example.test?secret=key",
        "https://example.test#fragment",
        "https://example.test\n",
        "https://example.test\\@evil.test",
        "https://example.test:bad",
    ],
)
def test_bad_urls(url):
    with pytest.raises(ValueError):
        SystemOneBackend(base_url=url, model="m")


@pytest.mark.parametrize("endpoint", ["//evil.test/api", "https://evil.test", "/api?secret=x"])
def test_endpoint_cannot_redirect_host(endpoint):
    with pytest.raises(ValueError):
        SystemOneBackend(base_url="https://example.test", model="m", endpoint=endpoint)


def test_kev_never_inherits_jev_credentials(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "private")
    monkeypatch.setenv("DEAVL_PROVIDER", "kev")
    backend = load_backend()
    assert backend.base_url == "http://127.0.0.1:8009"
    assert backend.auth == "none"
    assert "Authorization" not in backend._headers()
    assert "private" not in str(backend.identity())


@pytest.mark.parametrize(
    "provider,auth,keyenv,header",
    [
        ("jev", "bearer", "JEV_API_KEY", "Authorization"),
        ("azure", "api_key", "AZURE_API_KEY", "api-key"),
        ("azure", "bearer", "AZURE_ENTRA_TOKEN", "Authorization"),
        ("kev", "bearer", "KEV_API_KEY", "Authorization"),
    ],
)
def test_provider_presets_auth(monkeypatch, provider, auth, keyenv, header):
    monkeypatch.setenv(keyenv, "key")
    backend = load_backend(
        {
            "provider": provider,
            "base_url": "https://example.test",
            "model": "m",
            "auth": auth,
            "api_key_env": keyenv,
        }
    )
    assert backend._headers()[header] == ("key" if auth == "api_key" else "Bearer key")
    assert backend.endpoint == (
        "/providers/microsoft/v1/systemone" if provider == "azure" else "/v1/systemone"
    )


def test_entra_refresh(monkeypatch):
    calls = []

    class Credential:
        def get_token(self, scope):
            calls.append(scope)
            return types.SimpleNamespace(token=f"token-{len(calls)}")

    module = types.ModuleType("azure.identity")
    module.DefaultAzureCredential = Credential
    monkeypatch.setitem(sys.modules, "azure.identity", module)
    backend = load_backend(
        {
            "provider": "azure",
            "auth": "entra",
            "model": "deployment",
            "base_url": "https://example.test",
        }
    )
    assert backend._headers()["Authorization"] == "Bearer token-1"
    assert backend._headers()["Authorization"] == "Bearer token-2"
    assert calls == ["https://cognitiveservices.azure.com/.default"] * 2


@pytest.mark.parametrize(
    "data",
    [
        {"provider": "unknown"},
        {"provider": "azure"},
        {"api_key": "do-not-echo"},
        {"provider": "kev", "auth": "entra"},
        {"provider": "kev", "auth": "bearer"},
        {"provider": "kev", "auth": "none", "api_key_env": "KEY"},
        {"provider": "kev", "kinds": ["chat"]},
        {"provider": "kev", "max_questions": False},
        {"provider": "kev", "timeout": -1},
        {"provider": "kev", "allow_insecure_http": "false"},
    ],
)
def test_bad_config(data):
    with pytest.raises(ValueError) as exc:
        load_backend(data)
    assert "do-not-echo" not in str(exc.value)


def test_file_precedence(tmp_path, monkeypatch):
    path = tmp_path / "decision.json"
    path.write_text('{"provider":"kev"}', encoding="utf-8")
    monkeypatch.setenv("DEAVL_CONFIG", str(path))
    monkeypatch.setenv("DEAVL_PROVIDER", "unknown")
    assert load_backend().provider == "kev"
    assert load_backend(BackendConfig(provider="kev")).provider == "kev"


@pytest.fixture
def server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append((self.path, dict(self.headers), json.loads(data)))
            if self.path.startswith("/redirect/"):
                self.send_response(int(self.path.rsplit("/", 1)[1]))
                self.send_header("Location", "/stolen")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response()).encode())

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", calls
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def test_real_http_batches_all_primitives(server):
    url, calls = server
    backend = load_backend({"provider": "kev", "base_url": url})
    batch = backend.decide({"input": "text"}, QUESTIONS)
    assert len(calls) == 1
    assert set(calls[0][2]["questions"]) == {"yes", "label", "grade"}
    assert calls[0][2]["state"] == {"input": "text"}
    assert len(batch.answers) == 3


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_are_never_followed(server, status):
    url, calls = server
    backend = SystemOneBackend(
        base_url=url,
        model="m",
        endpoint=f"/redirect/{status}",
        auth="bearer",
        credential="private-key",
    )
    with pytest.raises(RuntimeError, match=str(status)):
        backend.decide("private prompt", QUESTIONS)
    assert len(calls) == 1


def test_error_bodies_are_not_exposed(monkeypatch):
    backend = load_backend({"provider": "kev"})

    def fail(req):
        raise HTTPError(req.full_url, 401, "private prompt", {}, io.BytesIO(b"private-key"))

    monkeypatch.setattr(backend, "_open", fail)
    with pytest.raises(RuntimeError) as exc:
        backend.decide("private prompt", QUESTIONS)
    assert "private" not in str(exc.value)
