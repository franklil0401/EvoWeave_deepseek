"""Runtime limits and observed resource usage."""

from pydantic import Field, model_validator

from evoweave_ds.domain.base import DomainModel


class RuntimeLimits(DomainModel):
    """Deterministic stop conditions; these are not monetary budgets."""

    max_steps: int = Field(default=64, ge=1, le=10_000)
    max_input_tokens: int = Field(default=1_000_000, ge=1)
    max_output_tokens: int = Field(default=16_384, ge=1)
    max_total_output_tokens: int = Field(default=262_144, ge=1)
    max_tool_calls: int = Field(default=64, ge=0, le=10_000)
    timeout_seconds: int = Field(default=1_800, ge=1, le=86_400)


class ResourceUsage(DomainModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)
    steps: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_token_breakdown(self) -> "ResourceUsage":
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("缓存命中输入 Token 不能超过总输入 Token")
        return self

    def exceeds(self, limits: RuntimeLimits) -> bool:
        """Return whether observed usage violates any deterministic limit."""

        return (
            self.input_tokens > limits.max_input_tokens
            or self.output_tokens + self.reasoning_tokens > limits.max_total_output_tokens
            or self.elapsed_ms > limits.timeout_seconds * 1_000
            or self.steps > limits.max_steps
            or self.tool_calls > limits.max_tool_calls
        )
