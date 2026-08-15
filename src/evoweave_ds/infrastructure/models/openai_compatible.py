"""Minimal OpenAI-compatible HTTP gateway for the three configured providers."""

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import Field, ValidationInfo, field_validator

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import InputModality, ModelAvailability, ModelTier
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.model_routing import ModelProfile
from evoweave_ds.domain.ports import ModelRequest, ModelResponse, ModelToolCall


class ProviderConfig(DomainModel):
    provider: str = Field(pattern=r"^[a-z0-9_-]+$")
    base_url: str = Field(min_length=8, max_length=2_048)
    api_key_env: str = Field(min_length=1, max_length=128)
    profiles: tuple[ModelProfile, ...] = Field(min_length=1)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith("https://"):
            raise ValueError("模型 Base URL 必须使用 HTTPS")
        return normalized

    @field_validator("profiles")
    @classmethod
    def validate_profiles(
        cls,
        values: tuple[ModelProfile, ...],
        info: ValidationInfo,
    ) -> tuple[ModelProfile, ...]:
        provider = info.data.get("provider")
        if provider is not None and any(item.provider != provider for item in values):
            raise ValueError("模型 Profile 必须属于 ProviderConfig.provider")
        return values


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes


class HttpTransport(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> HttpResponse: ...


class UrllibHttpTransport:
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> HttpResponse:
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return HttpResponse(status=response.status, body=response.read())
        except urllib.error.HTTPError as exc:
            return HttpResponse(status=exc.code, body=exc.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DomainError(
                ErrorCode.MODEL_UNAVAILABLE,
                "模型服务网络请求失败",
                details={"exception_type": type(exc).__name__},
            ) from exc


class OpenAICompatibleModelGateway:
    def __init__(
        self,
        providers: tuple[ProviderConfig, ...],
        *,
        transport: HttpTransport | None = None,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        if len({item.provider for item in providers}) != len(providers):
            raise ValueError("provider 不能重复")
        self._providers = {item.provider: item for item in providers}
        self._transport = transport or UrllibHttpTransport()
        self._environment = environment if environment is not None else os.environ
        self._timeout_seconds = timeout_seconds

    def list_profiles(self) -> tuple[ModelProfile, ...]:
        return tuple(
            profile
            for provider in sorted(self._providers)
            for profile in self._providers[provider].profiles
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        provider, model_id = _split_model_key(request.model_key)
        config = self._provider(provider)
        key = self._api_key(config)
        messages: list[dict[str, object]] = [{"role": "system", "content": request.messages[0]}]
        user_text = "\n\n".join(request.messages[1:]) or request.messages[0]
        messages.append({"role": "user", "content": user_text})
        payload_dict: dict[str, object] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "temperature": 0,
            "reasoning_effort": request.reasoning_effort,
        }
        if request.tools:
            payload_dict["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        payload = json.dumps(
            payload_dict,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        response = self._transport.request(
            method="POST",
            url=f"{config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            body=payload,
            timeout_seconds=self._timeout_seconds,
        )
        data = _response_json(response)
        try:
            message = data["choices"][0]["message"]
            usage = data.get("usage", {})
            details = usage.get("completion_tokens_details", {})
            tool_calls = _parse_tool_calls(message.get("tool_calls"))
            return ModelResponse(
                model_key=request.model_key,
                text=_content_text(message.get("content")),
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                reasoning_tokens=int(details.get("reasoning_tokens", 0)),
                tool_calls=tool_calls,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型响应结构无效") from exc

    def discover_model_ids(self, provider: str) -> tuple[str, ...]:
        config = self._provider(provider)
        response = self._transport.request(
            method="GET",
            url=f"{config.base_url}/models",
            headers={"Authorization": f"Bearer {self._api_key(config)}"},
            body=None,
            timeout_seconds=self._timeout_seconds,
        )
        data = _response_json(response)
        try:
            identifiers = {
                str(item["id"])
                for item in data["data"]
                if isinstance(item, dict) and item.get("id")
            }
        except (KeyError, TypeError) as exc:
            raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型列表响应结构无效") from exc
        return tuple(sorted(identifiers))

    def available_profiles(self, provider: str) -> tuple[ModelProfile, ...]:
        discovered = set(self.discover_model_ids(provider))
        checked_at = datetime.now(UTC)
        return tuple(
            profile.model_copy(
                update={
                    "availability": (
                        ModelAvailability.AVAILABLE
                        if profile.model_id in discovered
                        else ModelAvailability.UNAVAILABLE
                    ),
                    "checked_at": checked_at if profile.model_id in discovered else None,
                }
            )
            for profile in self._provider(provider).profiles
        )

    def _provider(self, provider: str) -> ProviderConfig:
        try:
            return self._providers[provider]
        except KeyError as exc:
            raise DomainError(ErrorCode.INVALID_SPEC, f"未知模型供应商：{provider}") from exc

    def _api_key(self, config: ProviderConfig) -> str:
        key = self._environment.get(config.api_key_env, "").strip()
        if not key:
            raise DomainError(
                ErrorCode.MODEL_UNAVAILABLE,
                f"缺少环境变量 {config.api_key_env}",
            )
        return key


def default_provider_configs() -> tuple[ProviderConfig, ...]:
    return (
        ProviderConfig(
            provider="deepseek",
            base_url="https://api.deepseek.com",
            api_key_env="Deepseek_api_key",
            profiles=(_profile("deepseek", "deepseek-v4-flash", ModelTier.LOW),),
        ),
    )


def _profile(
    provider: str,
    model_id: str,
    tier: ModelTier,
) -> ModelProfile:
    return ModelProfile(
        provider=provider,
        model_id=model_id,
        tier=tier,
        availability=ModelAvailability.UNKNOWN,
        input_modalities=(InputModality.TEXT,),
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_thinking=True,
    )


def _split_model_key(model_key: str) -> tuple[str, str]:
    provider, separator, model_id = model_key.partition(":")
    if not separator or not provider or not model_id:
        raise DomainError(ErrorCode.INVALID_SPEC, "模型 key 必须为 provider:model_id")
    return provider, model_id


def _response_json(response: HttpResponse) -> dict[str, Any]:
    if not 200 <= response.status < 300:
        provider_error_code = _provider_error_code(response.body)
        if response.status == 400:
            code = ErrorCode.MODEL_CAPABILITY_MISMATCH
            message = "模型请求参数不兼容"
        else:
            code = ErrorCode.MODEL_UNAVAILABLE
            message = "模型服务返回非成功状态"
        raise DomainError(
            code,
            message,
            details={
                "http_status": response.status,
                "provider_error_code": provider_error_code,
            },
        )
    try:
        data = json.loads(response.body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型服务返回非 JSON") from exc
    if not isinstance(data, dict):
        raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型服务 JSON 顶层必须为对象")
    return data


def _provider_error_code(body: bytes) -> str | None:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("error"), dict):
        return None
    value = data["error"].get("code")
    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    return value


def _parse_tool_calls(value: object) -> tuple[ModelToolCall, ...]:
    """Parse structured tool calls from an OpenAI-compatible response."""

    if value is None:
        return ()
    if not isinstance(value, list):
        raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型 tool_calls 结构无效")
    calls: list[ModelToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型 tool_calls 结构无效")
        function = item.get("function")
        if not isinstance(function, dict):
            raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型 tool_calls 结构无效")
        name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(name, str) or not isinstance(raw_arguments, str):
            raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型 tool_calls 结构无效")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise DomainError(
                ErrorCode.INVALID_MODEL_OUTPUT, "模型 tool_calls 参数不是 JSON"
            ) from exc
        if not isinstance(arguments, dict):
            raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型 tool_calls 参数不是对象")
        calls.append(ModelToolCall(name=name, arguments=arguments))
    return tuple(calls)


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = [
            item.get("text", "")
            for item in value
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if texts:
            return "".join(texts)
    raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "模型响应 content 不是文本")
