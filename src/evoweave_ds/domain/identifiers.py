"""Strong identifiers used across runs, tasks, agents, and artifacts."""

import re
from typing import ClassVar, Self
from uuid import uuid4

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

_BODY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{5,127}$")


class Identifier(str):
    """A validated string identifier with a type-specific prefix."""

    prefix: ClassVar[str] = "id"

    def __new__(cls, value: str) -> Self:
        if not isinstance(value, str):
            raise TypeError("标识必须是字符串")
        expected_prefix = f"{cls.prefix}_"
        if not value.startswith(expected_prefix):
            raise ValueError(f"标识必须以 {expected_prefix!r} 开头")
        body = value[len(expected_prefix) :]
        if not _BODY_PATTERN.fullmatch(body):
            raise ValueError("标识主体必须为 6-128 位小写字母、数字、下划线或连字符")
        return str.__new__(cls, value)

    @classmethod
    def new(cls) -> Self:
        """Create a random identifier with the correct prefix."""

        return cls(f"{cls.prefix}_{uuid4().hex}")

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: object,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )


class RunId(Identifier):
    prefix = "run"


class TaskId(Identifier):
    prefix = "task"


class AgentId(Identifier):
    prefix = "agent"


class ArtifactId(Identifier):
    prefix = "artifact"


class EvidenceId(Identifier):
    prefix = "evidence"


class SpecId(Identifier):
    prefix = "spec"


class EventId(Identifier):
    prefix = "event"


class GraphId(Identifier):
    prefix = "graph"


class WorkspaceId(Identifier):
    prefix = "workspace"


class TaskLeaseId(Identifier):
    prefix = "lease"


class IntegrationId(Identifier):
    prefix = "integration"
