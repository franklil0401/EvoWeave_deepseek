import json
from collections.abc import Mapping

import pytest

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.ports import ModelRequest
from evoweave_ds.infrastructure.models.doctor import ModelDoctor
from evoweave_ds.infrastructure.models.openai_compatible import (
    HttpResponse,
    OpenAICompatibleModelGateway,
    default_provider_configs,
)


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )
        if url.endswith("/models"):
            return HttpResponse(
                status=200,
                body=json.dumps({"data": [{"id": "deepseek-v4-flash"}, {"id": "other"}]}).encode(),
            )
        return HttpResponse(
            status=200,
            body=json.dumps(
                {
                    "choices": [{"message": {"content": '{"action":"finish"}'}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "completion_tokens_details": {"reasoning_tokens": 1},
                    },
                }
            ).encode(),
        )


def test_gateway_sends_openai_compatible_request_and_parses_usage() -> None:
    transport = FakeTransport()
    providers = default_provider_configs()
    gateway = OpenAICompatibleModelGateway(
        providers,
        transport=transport,
        environment={"Deepseek_api_key": "secret-for-test"},
    )

    response = gateway.complete(
        ModelRequest(
            model_key="deepseek:deepseek-v4-flash",
            messages=("system", "user"),
            max_output_tokens=100,
        )
    )

    assert response.text == '{"action":"finish"}'
    assert (response.input_tokens, response.output_tokens, response.reasoning_tokens) == (10, 3, 1)
    call = transport.calls[0]
    assert call["url"] == "https://api.deepseek.com/chat/completions"
    payload = json.loads(call["body"])
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["temperature"] == 0


def test_doctor_never_requires_network_for_key_presence_check() -> None:
    transport = FakeTransport()
    providers = default_provider_configs()
    environment = {"Deepseek_api_key": "secret-for-test"}
    gateway = OpenAICompatibleModelGateway(
        providers,
        transport=transport,
        environment=environment,
    )

    results = ModelDoctor(providers, gateway, environment=environment).inspect(network=False)

    assert results[0].provider == "deepseek"
    assert results[0].key_present is True
    assert all(item.network_checked is False for item in results)
    assert transport.calls == []


def test_network_doctor_lists_models_without_completions() -> None:
    transport = FakeTransport()
    provider = default_provider_configs()[0]
    environment = {"Deepseek_api_key": "secret-for-test"}
    gateway = OpenAICompatibleModelGateway(
        (provider,),
        transport=transport,
        environment=environment,
    )

    result = ModelDoctor((provider,), gateway, environment=environment).inspect(network=True)[0]

    assert result.reachable is True
    assert result.discovered_model_ids == ("deepseek-v4-flash", "other")
    assert [item["method"] for item in transport.calls] == ["GET"]


def test_gateway_classifies_bad_request_without_provider_message() -> None:
    class BadRequestTransport(FakeTransport):
        def request(
            self,
            *,
            method: str,
            url: str,
            headers: Mapping[str, str],
            body: bytes | None,
            timeout_seconds: int,
        ) -> HttpResponse:
            return HttpResponse(
                status=400,
                body=json.dumps(
                    {
                        "error": {
                            "code": "invalid_parameter_error",
                            "message": "provider message must not be persisted",
                        }
                    }
                ).encode(),
            )

    gateway = OpenAICompatibleModelGateway(
        default_provider_configs(),
        transport=BadRequestTransport(),
        environment={"Deepseek_api_key": "secret-for-test"},
    )

    with pytest.raises(DomainError) as error:
        gateway.complete(
            ModelRequest(
                model_key="deepseek:deepseek-v4-flash",
                messages=("system", "user"),
                max_output_tokens=64,
            )
        )

    assert error.value.code is ErrorCode.MODEL_CAPABILITY_MISMATCH
    assert error.value.details == {
        "http_status": 400,
        "provider_error_code": "invalid_parameter_error",
    }
