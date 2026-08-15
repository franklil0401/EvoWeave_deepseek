"""Reusable validation helpers for repository-bound contracts."""

from pathlib import PurePosixPath


def validate_repository_path(value: str) -> str:
    """Validate a portable path relative to the repository root."""

    if not value:
        raise ValueError("仓库路径不能为空")
    if "\\" in value:
        raise ValueError("仓库路径必须使用正斜杠")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise ValueError("仓库路径不能是绝对路径")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("仓库路径不能包含空段、. 或 ..")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError("仓库路径必须是规范化的 POSIX 相对路径")
    return value


def validate_unique_strings(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} 不能包含重复项")
    return values


def path_is_within_scopes(path: str, scopes: tuple[str, ...]) -> bool:
    """Return whether a repository path is equal to or below an allowed scope."""

    return any(path == scope or path.startswith(f"{scope}/") for scope in scopes)


def validate_scope_subset(
    child_scopes: tuple[str, ...],
    parent_scopes: tuple[str, ...],
    *,
    child_name: str,
    parent_name: str,
) -> None:
    uncovered = [scope for scope in child_scopes if not path_is_within_scopes(scope, parent_scopes)]
    if uncovered:
        raise ValueError(f"{child_name} 必须被 {parent_name} 覆盖：{uncovered}")
