from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .backends import SystemOneBackend
from .jsonio import read_json
from .types import Capabilities, nonempty


@dataclass(frozen=True)
class BackendConfig:
    provider: str = "systemone"
    model: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    auth: str | None = None
    api_key_env: str | None = None
    timeout: float = 30
    allow_insecure_http: bool = False
    kinds: tuple[str, ...] = ("noul", "choice", "score")
    max_state_bytes: int = 1_048_576
    max_request_bytes: int = 2_097_152
    max_questions: int = 128
    max_choice_options: int = 255
    max_score_levels: int = 10
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None

    def __post_init__(self):
        if not isinstance(self.provider, str) or self.provider not in {
            "jev",
            "kev",
            "azure",
            "systemone",
        }:
            raise ValueError("provider must be jev, kev, azure, or systemone.")
        for name in ("model", "base_url", "endpoint", "auth", "api_key_env"):
            if getattr(self, name) is not None:
                nonempty(getattr(self, name), name)
        if self.auth is not None and self.auth not in {"none", "bearer", "api_key", "entra"}:
            raise ValueError("Unsupported authentication method.")
        if self.api_key_env is not None and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must name an environment variable, not a secret.")
        Capabilities(
            frozenset(self.kinds),
            True,
            self.max_state_bytes,
            self.max_request_bytes,
            self.max_questions,
            self.max_choice_options,
            self.max_score_levels,
        )

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError("Invalid/unknown backend configuration fields.")
        return cls(**data)

    @classmethod
    def from_env(cls):
        data = {}
        for item in fields(cls):
            key = "DEAVL_" + item.name.upper()
            if key not in os.environ:
                continue
            value: Any = os.environ[key]
            if item.name == "allow_insecure_http":
                if value.lower() not in {"true", "false"}:
                    raise ValueError("DEAVL_ALLOW_INSECURE_HTTP must be true or false.")
                value = value.lower() == "true"
            elif item.name == "kinds":
                value = value.split(",")
            elif item.name.startswith("max_"):
                value = int(value)
            elif item.name == "timeout" or item.name.endswith("per_million"):
                value = float(value)
            data[item.name] = value
        return cls.from_dict(data)

    def build(self):
        presets = {
            "jev": ("https://api.typesafe.ai", "jev-1.13.0"),
            "kev": ("http://127.0.0.1:8009", "kev-latest"),
        }
        default_url, default_model = presets.get(self.provider, (None, None))
        base_url, model = self.base_url or default_url, self.model or default_model
        if base_url is None or model is None:
            raise ValueError(
                "Explicit base_url and model/deployment are required for this provider."
            )
        auth = self.auth or (
            "api_key"
            if self.provider == "azure"
            else "bearer"
            if self.provider == "jev" or self.api_key_env
            else "none"
        )
        if self.provider == "jev" and auth != "bearer":
            raise ValueError("The Jev preset requires bearer authentication.")
        if self.provider == "azure" and auth == "none":
            raise ValueError("Azure requires authentication.")
        if auth == "entra" and self.provider != "azure":
            raise ValueError("Entra authentication is supported only for Azure.")
        if auth in {"none", "entra"} and self.api_key_env:
            raise ValueError("This auth method cannot use api_key_env.")
        credential = token_provider = None
        if auth in {"bearer", "api_key"}:
            env = self.api_key_env
            if env is None and self.provider == "jev":
                env = "JEV_API_KEY" if os.environ.get("JEV_API_KEY") else "TYPESAFE_API_KEY"
            if env is None and self.provider == "azure":
                env = "AZURE_API_KEY" if auth == "api_key" else "AZURE_ENTRA_TOKEN"
            if env is None:
                raise ValueError("Explicit api_key_env is required for this server.")
            credential = os.environ.get(env)
            if not credential:
                raise ValueError(f"Set the credential environment variable {env}.")
        if auth == "entra":
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise RuntimeError("Install DEavl[azure] for Entra authentication.") from None
            identity = DefaultAzureCredential()

            def token_provider():
                return identity.get_token("https://cognitiveservices.azure.com/.default").token

        return SystemOneBackend(
            base_url=base_url,
            model=model,
            provider=self.provider,
            endpoint=self.endpoint
            or (
                "/providers/microsoft/v1/systemone" if self.provider == "azure" else "/v1/systemone"
            ),
            auth="bearer" if auth == "entra" else auth,
            credential=credential,
            token_provider=token_provider,
            timeout=self.timeout,
            allow_insecure_http=self.allow_insecure_http,
            capabilities=Capabilities(
                frozenset(self.kinds),
                True,
                self.max_state_bytes,
                self.max_request_bytes,
                self.max_questions,
                self.max_choice_options,
                self.max_score_levels,
            ),
            input_usd_per_million=self.input_usd_per_million,
            output_usd_per_million=self.output_usd_per_million,
        )

    def to_dict(self):
        return asdict(self)


def load_backend(config: BackendConfig | Mapping | str | Path | None = None) -> SystemOneBackend:
    if config is None:
        path = os.environ.get("DEAVL_CONFIG")
        config = BackendConfig.from_dict(read_json(path)) if path else BackendConfig.from_env()
    elif isinstance(config, (str, Path)):
        config = BackendConfig.from_dict(read_json(config))
    elif isinstance(config, Mapping):
        config = BackendConfig.from_dict(config)
    if not isinstance(config, BackendConfig):
        raise ValueError("Provide BackendConfig, a mapping, or a JSON file path.")
    return config.build()
