"""Discover deterministic validation commands from committed project metadata."""

import tomllib
from collections.abc import Mapping

from evoweave_ds.domain.repository_models import RepositoryFile, ValidationCommand
from evoweave_ds.repository.git_inspector import GitInspector


class ValidationCommandDiscoverer:
    def discover(
        self,
        *,
        inspector: GitInspector,
        files: tuple[RepositoryFile, ...],
    ) -> tuple[ValidationCommand, ...]:
        paths = {item.path for item in files}
        pyproject = _read_pyproject(inspector) if "pyproject.toml" in paths else {}
        commands: list[ValidationCommand] = []
        dependency_names = _dependency_names(pyproject)
        has_tests = any(item.kind == "test" and item.path.endswith(".py") for item in files)
        tool_value = pyproject.get("tool")
        tool: Mapping[str, object] = tool_value if isinstance(tool_value, Mapping) else {}

        if has_tests or "pytest" in dependency_names or "pytest" in tool:
            commands.append(
                ValidationCommand(
                    command_id="pytest",
                    argv=("python", "-m", "pytest", "-q", "-p", "no:cacheprovider"),
                    source="pytest",
                )
            )
        if "ruff" in dependency_names or "ruff" in tool or "ruff.toml" in paths:
            commands.append(
                ValidationCommand(
                    command_id="ruff",
                    argv=("python", "-m", "ruff", "check", ".", "--no-cache"),
                    source="ruff",
                )
            )
        commands.extend(_project_commands(tool))
        unique = {command.command_id: command for command in commands}
        return tuple(unique[key] for key in sorted(unique))


def _read_pyproject(inspector: GitInspector) -> dict[str, object]:
    try:
        parsed = tomllib.loads(inspector.read_bytes("pyproject.toml").decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}
    return dict(parsed)


def _dependency_names(document: Mapping[str, object]) -> set[str]:
    values: list[str] = []
    project = document.get("project")
    if isinstance(project, Mapping):
        dependencies = project.get("dependencies")
        if isinstance(dependencies, list):
            values.extend(item for item in dependencies if isinstance(item, str))
        optional = project.get("optional-dependencies")
        if isinstance(optional, Mapping):
            for group in optional.values():
                if isinstance(group, list):
                    values.extend(item for item in group if isinstance(item, str))
    dependency_groups = document.get("dependency-groups")
    if isinstance(dependency_groups, Mapping):
        for group in dependency_groups.values():
            if isinstance(group, list):
                values.extend(item for item in group if isinstance(item, str))
    return {_dependency_name(value) for value in values}


def _dependency_name(requirement: str) -> str:
    normalized = requirement.split(";", 1)[0].strip().lower()
    for separator in ("[", "<", ">", "=", "!", "~", " "):
        normalized = normalized.split(separator, 1)[0]
    return normalized.replace("_", "-")


def _project_commands(tool: object) -> list[ValidationCommand]:
    if not isinstance(tool, Mapping):
        return []
    evoweave_ds = tool.get("evoweave_ds")
    if not isinstance(evoweave_ds, Mapping):
        return []
    validation = evoweave_ds.get("validation")
    if not isinstance(validation, Mapping):
        return []
    configured = validation.get("commands")
    if not isinstance(configured, Mapping):
        return []
    commands: list[ValidationCommand] = []
    for command_id, argv in sorted(configured.items(), key=lambda item: str(item[0])):
        if not isinstance(command_id, str) or not isinstance(argv, list):
            continue
        if not argv or not all(isinstance(argument, str) and argument for argument in argv):
            continue
        commands.append(
            ValidationCommand(
                command_id=command_id,
                argv=tuple(argv),
                source="project",
            )
        )
    return commands
