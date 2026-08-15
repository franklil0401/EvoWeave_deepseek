"""Deterministic runtime limit tracking; no monetary budget is involved."""

from collections.abc import Callable
from time import monotonic

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.ports import ModelResponse
from evoweave_ds.domain.resources import ResourceUsage, RuntimeLimits


class RuntimeLimitTracker:
    def __init__(
        self,
        limits: RuntimeLimits,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._limits = limits
        self._clock = clock
        self._started_at = clock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._reasoning_tokens = 0
        self._steps = 0
        self._tool_calls = 0

    def record_step(self) -> None:
        self._steps += 1
        self.assert_within_limits()

    def record_model_response(self, response: ModelResponse) -> None:
        self._input_tokens += response.input_tokens
        self._output_tokens += response.output_tokens
        self._reasoning_tokens += response.reasoning_tokens
        self.assert_within_limits()

    def record_tool_call(self) -> None:
        self._tool_calls += 1
        self.assert_within_limits()

    def usage(self) -> ResourceUsage:
        elapsed_ms = max(0, int((self._clock() - self._started_at) * 1_000))
        return ResourceUsage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            reasoning_tokens=self._reasoning_tokens,
            elapsed_ms=elapsed_ms,
            steps=self._steps,
            tool_calls=self._tool_calls,
        )

    def assert_within_limits(self) -> None:
        usage = self.usage()
        if usage.exceeds(self._limits):
            raise DomainError(
                ErrorCode.RUNTIME_LIMIT_EXCEEDED,
                "Worker 已达到确定性运行上限",
                details=usage.model_dump(mode="json"),
            )
