"""Deterministic command authorization with no shell interpretation."""

from pathlib import PurePath

from evoweave_ds.domain.errors import DomainError, ErrorCode

_FORBIDDEN_FRAGMENTS = ("&&", "||", ";", "\n", "\r", ">", "<", "`", "$(")


class CommandPolicy:
    def authorize(
        self,
        argv: tuple[str, ...],
        *,
        allowed_commands: tuple[str, ...],
    ) -> None:
        if not argv or not argv[0].strip():
            raise DomainError(ErrorCode.COMMAND_DENIED, "命令参数不能为空")
        if "/" in argv[0] or "\\" in argv[0]:
            raise DomainError(ErrorCode.COMMAND_DENIED, "命令必须使用已授权的可执行文件名")
        executable = PurePath(argv[0]).name.lower()
        if any("/" in item or "\\" in item for item in allowed_commands):
            raise DomainError(ErrorCode.COMMAND_DENIED, "命令白名单不能包含路径")
        allowed = {PurePath(item).name.lower() for item in allowed_commands}
        if executable not in allowed:
            raise DomainError(
                ErrorCode.COMMAND_DENIED,
                f"命令未被执行规格授权：{executable}",
            )
        for argument in argv:
            if "\x00" in argument or any(fragment in argument for fragment in _FORBIDDEN_FRAGMENTS):
                raise DomainError(
                    ErrorCode.COMMAND_DENIED,
                    "命令参数包含禁止的 shell 控制字符",
                )
