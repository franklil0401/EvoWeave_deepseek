"""Build deterministic internal module dependencies from extracted imports."""

from collections import defaultdict, deque

from evoweave_ds.domain.repository_models import ModuleDependency, PythonImport, RepositoryFile


class DependencyGraphBuilder:
    def build(
        self,
        *,
        files: tuple[RepositoryFile, ...],
        imports: tuple[PythonImport, ...],
    ) -> tuple[ModuleDependency, ...]:
        known_modules = {item.module_name for item in files if item.module_name is not None}
        dependencies: dict[tuple[str, str], ModuleDependency] = {}
        for item in imports:
            target = _resolve_known_target(item, known_modules)
            if target is None or target == item.importer_module:
                continue
            key = (item.importer_module, target)
            candidate = ModuleDependency(
                importer_module=item.importer_module,
                imported_module=target,
                path=item.path,
                line=item.line,
                evidence_id=item.evidence_id,
            )
            current = dependencies.get(key)
            if current is None or (candidate.path, candidate.line) < (current.path, current.line):
                dependencies[key] = candidate
        return tuple(
            sorted(
                dependencies.values(),
                key=lambda item: (item.importer_module, item.imported_module),
            )
        )


def dependency_neighbors(
    modules: set[str],
    dependencies: tuple[ModuleDependency, ...],
    *,
    max_depth: int = 1,
) -> dict[str, int]:
    """Return both importers and imports around seed modules with shortest depth."""

    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in dependencies:
        adjacency[edge.importer_module].add(edge.imported_module)
        adjacency[edge.imported_module].add(edge.importer_module)
    distances = {module: 0 for module in modules}
    queue = deque(sorted(modules))
    while queue:
        module = queue.popleft()
        depth = distances[module]
        if depth >= max_depth:
            continue
        for neighbor in sorted(adjacency[module]):
            if neighbor not in distances:
                distances[neighbor] = depth + 1
                queue.append(neighbor)
    return distances


def maximum_dependency_fan_out(dependencies: tuple[ModuleDependency, ...]) -> int:
    fan_out: dict[str, set[str]] = defaultdict(set)
    for edge in dependencies:
        fan_out[edge.importer_module].add(edge.imported_module)
    return max((len(targets) for targets in fan_out.values()), default=0)


def _resolve_known_target(item: PythonImport, known_modules: set[str]) -> str | None:
    candidates: list[str] = []
    if item.imported_name and item.imported_name != "*":
        candidates.append(f"{item.imported_module}.{item.imported_name}")
    candidates.append(item.imported_module)
    for candidate in candidates:
        if candidate in known_modules:
            return candidate
    for candidate in candidates:
        parts = candidate.split(".")
        while len(parts) > 1:
            parts.pop()
            parent = ".".join(parts)
            if parent in known_modules:
                return parent
    return None
