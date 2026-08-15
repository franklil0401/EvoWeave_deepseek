"""Read-only provider configuration and optional model-discovery diagnostics."""

import os
from collections.abc import Mapping
from datetime import datetime

from pydantic import Field

from evoweave_ds.domain.base import DomainModel, utc_now
from evoweave_ds.domain.errors import DomainError
from evoweave_ds.infrastructure.models.openai_compatible import (
    OpenAICompatibleModelGateway,
    ProviderConfig,
)


class ProviderDoctorResult(DomainModel):
    provider: str
    base_url: str
    api_key_env: str
    key_present: bool
    network_checked: bool
    reachable: bool | None = None
    discovered_model_ids: tuple[str, ...] = ()
    error_code: str | None = Field(default=None, max_length=128)
    checked_at: datetime = Field(default_factory=utc_now)


class ModelDoctor:
    def __init__(
        self,
        providers: tuple[ProviderConfig, ...],
        gateway: OpenAICompatibleModelGateway,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._providers = providers
        self._gateway = gateway
        self._environment = environment if environment is not None else os.environ

    def inspect(self, *, network: bool = False) -> tuple[ProviderDoctorResult, ...]:
        results: list[ProviderDoctorResult] = []
        for provider in self._providers:
            present = bool(self._environment.get(provider.api_key_env, "").strip())
            if not network or not present:
                results.append(
                    ProviderDoctorResult(
                        provider=provider.provider,
                        base_url=provider.base_url,
                        api_key_env=provider.api_key_env,
                        key_present=present,
                        network_checked=False,
                    )
                )
                continue
            try:
                model_ids = self._gateway.discover_model_ids(provider.provider)
            except DomainError as exc:
                results.append(
                    ProviderDoctorResult(
                        provider=provider.provider,
                        base_url=provider.base_url,
                        api_key_env=provider.api_key_env,
                        key_present=True,
                        network_checked=True,
                        reachable=False,
                        error_code=exc.code.value,
                    )
                )
            else:
                results.append(
                    ProviderDoctorResult(
                        provider=provider.provider,
                        base_url=provider.base_url,
                        api_key_env=provider.api_key_env,
                        key_present=True,
                        network_checked=True,
                        reachable=True,
                        discovered_model_ids=model_ids,
                    )
                )
        return tuple(results)
