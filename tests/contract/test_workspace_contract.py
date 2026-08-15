"""Contract tests for per-task workspace scope isolation."""

import pytest

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import TaskId
from evoweave_ds.domain.ports import WorkspaceAdapter
from evoweave_ds.infrastructure.workspaces.fake import FakeWorkspace


def test_workspace_reads_and_writes_inside_nested_scope() -> None:
    workspace = FakeWorkspace(
        task_id=TaskId.new(),
        files={"src/app.py": "old"},
        read_scope=("src",),
        write_scope=("src",),
    )
    assert isinstance(workspace, WorkspaceAdapter)
    assert workspace.read_text("src/app.py") == "old"
    workspace.write_text("src/app.py", "new")
    assert workspace.read_text("src/app.py") == "new"


def test_workspace_rejects_read_outside_scope() -> None:
    workspace = FakeWorkspace(
        task_id=TaskId.new(),
        files={"secrets.txt": "hidden"},
        read_scope=("src",),
    )
    with pytest.raises(DomainError) as error:
        workspace.read_text("secrets.txt")
    assert error.value.code is ErrorCode.WORKSPACE_ACCESS_DENIED


def test_workspace_rejects_path_traversal() -> None:
    workspace = FakeWorkspace(task_id=TaskId.new(), read_scope=("src",))
    with pytest.raises(DomainError) as error:
        workspace.read_text("src/../secrets.txt")
    assert error.value.code is ErrorCode.WORKSPACE_ACCESS_DENIED


def test_workspace_write_scope_must_be_covered_by_read_scope() -> None:
    with pytest.raises(ValueError, match="write_scope"):
        FakeWorkspace(
            task_id=TaskId.new(),
            read_scope=("src",),
            write_scope=("tests",),
        )
