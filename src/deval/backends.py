from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from ipaddress import ip_address
from typing import Any
from urllib import error, request
from urllib.parse import urlsplit

from .jsonio import canonical, fingerprint, loads, snapshot
from .types import Answer, Capabilities, DecisionBatch, Question, Usage, nonempty, number


class DecisionBackend(ABC):
    capabilities: Capabilities = Capabilities()

    @abstractmethod
    def decide(self, state: Any, questions: tuple[Question, ...]) -> DecisionBatch:
        raise NotImplementedError

    async def a_decide(self, state: Any, questions: tuple[Question, ...]) -> DecisionBatch:
        return await asyncio.to_thread(self.decide, state, questions)

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return public provider/model/config metadata. Never return credentials."""
        raise NotImplementedError


class FixtureBackend(DecisionBackend):
    """Explicit recorded answers for demos/tests. This is not inference."""

    def __init__(self, answers: dict[str, Answer], model: str = "fixture-not-inference"):
        self.answers = answers
        self.model = model

    def decide(self, state, questions):
        self.capabilities.check(state, questions)
        return DecisionBatch({q.id: self.answers[q.id] for q in questions}, self.model)

    def identity(self):
        return {"provider": "fixture", "model": self.model, "inference": False}


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class SystemOneBackend(DecisionBackend):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        endpoint: str = "/v1/systemone",
        auth: str = "none",
        credential: str | None = None,
        token_provider: Callable[[], str] | None = None,
        timeout: float = 30,
        capabilities: Capabilities | None = None,
        allow_insecure_http: bool = False,
        input_usd_per_million: float | None = None,
        output_usd_per_million: float | None = None,
        provider: str = "systemone",
    ):
        nonempty(base_url, "base_url")
        nonempty(endpoint, "endpoint")
        nonempty(model, "model")
        nonempty(provider, "provider")
        if type(allow_insecure_http) is not bool:
            raise ValueError("allow_insecure_http must be a boolean.")
        if any(char.isspace() or ord(char) < 32 for char in base_url + endpoint):
            raise ValueError("Decision URLs cannot contain whitespace/control characters.")
        if "\\" in base_url + endpoint:
            raise ValueError("Decision URLs cannot contain backslashes.")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Use an HTTP(S) base_url without credentials, query, or fragment.")
        parsed.port
        local = parsed.hostname == "localhost"
        try:
            local = local or ip_address(parsed.hostname).is_loopback
        except ValueError:
            pass
        if parsed.scheme == "http" and not local and not allow_insecure_http:
            raise ValueError("Use HTTPS outside loopback, or explicitly allow insecure HTTP.")
        path = urlsplit(endpoint)
        if path.scheme or path.netloc or path.query or path.fragment:
            raise ValueError("endpoint must be a relative API path.")
        if not isinstance(auth, str) or auth not in {"none", "bearer", "api_key"}:
            raise ValueError("auth must be none, bearer, or api_key.")
        if token_provider is not None and (
            not callable(token_provider) or auth != "bearer" or credential is not None
        ):
            raise ValueError("A token provider requires bearer auth without a static credential.")
        if auth == "none" and credential is not None:
            raise ValueError("A credential requires an authentication method.")
        if auth != "none" and token_provider is None:
            self._validate_credential(credential)
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else "/" + endpoint
        self.model, self.provider = model, provider
        self.auth, self._credential, self._token_provider = auth, credential, token_provider
        self.timeout = number(timeout, "timeout", low=0.001, high=3600)
        self.capabilities = capabilities or Capabilities()
        if not isinstance(self.capabilities, Capabilities):
            raise ValueError("capabilities must be Capabilities.")
        self.input_price = (
            None
            if input_usd_per_million is None
            else number(input_usd_per_million, "Input price", high=1e9)
        )
        self.output_price = (
            None
            if output_usd_per_million is None
            else number(output_usd_per_million, "Output price", high=1e9)
        )

    @staticmethod
    def _validate_credential(value):
        if (
            not isinstance(value, str)
            or not value
            or any(not 33 <= ord(char) <= 126 for char in value)
        ):
            raise ValueError("Credential must be non-empty printable ASCII without whitespace.")
        return value

    def identity(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "inference": True,
            "endpoint": self.base_url + self.endpoint,
            "auth": self.auth,
            "timeout": self.timeout,
            "capabilities": self.capabilities.to_dict(),
            "input_usd_per_million": self.input_price,
            "output_usd_per_million": self.output_price,
        }

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "DEval/0.2.0",
        }
        if self.auth != "none":
            try:
                key = self._token_provider() if self._token_provider else self._credential
            except Exception:
                raise RuntimeError("Decision credential acquisition failed.") from None
            key = self._validate_credential(key)
            headers["api-key" if self.auth == "api_key" else "Authorization"] = (
                key if self.auth == "api_key" else f"Bearer {key}"
            )
        return headers

    def decide(self, state, questions):
        questions = tuple(questions)
        if not questions or len({q.id for q in questions}) != len(questions):
            raise ValueError("Supply non-empty questions with unique IDs.")
        self.capabilities.check(state, questions)
        payload = {
            "model": self.model,
            "state": snapshot(state),
            "questions": {q.id: q.to_payload() for q in questions},
        }
        body = canonical(payload).encode("utf-8")
        if len(body) > self.capabilities.max_request_bytes:
            raise ValueError("Request exceeds the configured byte limit.")
        req = request.Request(
            self.base_url + self.endpoint, data=body, headers=self._headers(), method="POST"
        )
        try:
            with self._open(req) as response:
                raw = response.read(4_194_305)
                if len(raw) > 4_194_304:
                    raise ValueError("Decision response exceeds 4 MiB.")
                data = loads(raw)
        except error.HTTPError as exc:
            exc.close()
            raise RuntimeError(f"Decision backend HTTP {exc.code}.") from None
        except (error.URLError, OSError):
            raise RuntimeError("Decision backend request failed.") from None
        except (UnicodeDecodeError, ValueError):
            raise ValueError("Decision backend returned invalid or oversized JSON.") from None
        return self.parse(data, questions)

    def _open(self, req):
        return request.build_opener(_NoRedirect()).open(req, timeout=self.timeout)

    def parse(self, data: Any, questions: tuple[Question, ...]) -> DecisionBatch:
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise ValueError("A typed answers object is required.")
        if "truncated" in data and data["truncated"] is not False:
            raise ValueError("Input truncation is not acceptable for evaluation.")
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            raise ValueError("Usage must be an object.")
        if {"state_tokens", "state_tokens_used"} <= usage.keys():
            original, used = usage["state_tokens"], usage["state_tokens_used"]
            if any(type(v) is not int or v < 0 for v in (original, used)) or used != original:
                raise ValueError("The backend did not inspect the complete state.")
        if set(data["answers"]) != {q.id for q in questions}:
            raise ValueError("Missing or unexpected question IDs.")
        answers = {}
        for question in questions:
            raw = data["answers"][question.id]
            if not isinstance(raw, dict) or raw.get("type") != question.kind:
                raise ValueError("The decision answer has an incompatible type.")
            if question.kind == "noul":
                p = number(raw.get("noul"), "Noul probability")
                distribution = {"false": 1 - p, "true": p}
            else:
                distribution = raw.get("probabilities")
            answer = Answer(question.kind, distribution)
            answer.validate(question)
            # The distribution is authoritative; scalar/label disagreement is an error.
            if question.kind == "choice" and "choice" in raw:
                selected = raw["choice"]
                if selected not in answer.probabilities or answer.probabilities[selected] != max(
                    answer.probabilities.values()
                ):
                    raise ValueError("Choice label disagrees with its distribution.")
            if question.kind == "score" and "score" in raw:
                expected = sum(int(k) * p for k, p in answer.probabilities.items())
                scalar = number(raw["score"], "Ordinal score", high=len(question.options) - 1)
                if abs(scalar - expected) > 0.011:
                    raise ValueError("Ordinal score disagrees with its distribution.")
            answers[question.id] = answer
        counts = Usage(usage.get("input_tokens"), usage.get("output_tokens"))
        cost = None
        if (
            self.input_price is not None
            and self.output_price is not None
            and counts.input_tokens is not None
            and counts.output_tokens is not None
        ):
            cost = (
                counts.input_tokens * self.input_price + counts.output_tokens * self.output_price
            ) / 1e6
        batch = DecisionBatch(
            answers, data.get("model"), Usage(counts.input_tokens, counts.output_tokens, cost)
        )
        batch.validate(questions)
        return batch


def backend_fingerprint(backend: DecisionBackend) -> str:
    return fingerprint(backend.identity())
