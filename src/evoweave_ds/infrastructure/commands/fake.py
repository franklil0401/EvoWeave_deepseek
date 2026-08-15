"""Scripted command runner used by stage 1 tests."""

from collections.abc import Mapping

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.ports import CommandResult


class ScriptedCommandRunner:
    def __init__(self, results: Mapping[tuple[str, ...], CommandResult] | None = None) -> None:
        self._results = dict(results or {})
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def run(self, argv: tuple[str, ...], *, timeout_seconds: int) -> CommandResult:
        self.calls.append((argv, timeout_seconds))
        try:
            return self._results[argv]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.SCRIPT_EXHAUSTED,
                f"没有为命令配置脚本结果：{argv}",
            ) from exc
