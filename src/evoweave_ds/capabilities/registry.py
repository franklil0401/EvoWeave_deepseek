"""Registry for atomic capabilities; registration never grants permission."""

from collections.abc import Iterable

from evoweave_ds.capabilities.definitions import Capability, CapabilityDefinition
from evoweave_ds.domain.errors import DomainError, ErrorCode


class CapabilityRegistry:
    def __init__(self, capabilities: Iterable[Capability] = ()) -> None:
        self._capabilities: dict[str, Capability] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: Capability) -> None:
        name = capability.definition.name
        if name in self._capabilities:
            raise DomainError(
                ErrorCode.INVALID_SPEC,
                f"能力名称重复注册：{name}",
            )
        self._capabilities[name] = capability

    def get(self, name: str) -> Capability:
        try:
            return self._capabilities[name]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.CAPABILITY_NOT_FOUND,
                f"找不到原子能力：{name}",
            ) from exc

    def definitions(self) -> tuple[CapabilityDefinition, ...]:
        return tuple(self._capabilities[name].definition for name in sorted(self._capabilities))
