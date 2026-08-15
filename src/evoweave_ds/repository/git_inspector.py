"""Read a repository snapshot directly from immutable Git objects."""

import subprocess
from pathlib import Path

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.repository_models import GitRepositoryState, RepositoryBlob
from evoweave_ds.domain.validation import validate_repository_path


class GitInspector:
    """Resolve and read one commit without checking it out or changing the worktree."""

    def __init__(self, repository: Path | str, revision: str = "HEAD") -> None:
        requested_root = Path(repository).resolve()
        root_text = self._run_text(requested_root, "rev-parse", "--show-toplevel").strip()
        self._root = Path(root_text).resolve()
        self._commit = self._run_text(
            self._root,
            "rev-parse",
            "--verify",
            f"{revision}^{{commit}}",
        ).strip()
        if not _is_hex_object_id(self._commit):
            raise DomainError(
                ErrorCode.GIT_COMMAND_FAILED,
                "Git 返回了无效的 commit 标识",
                details={"revision": revision},
            )
        self._content_by_path: dict[str, bytes] = {}

    @property
    def repository_root(self) -> Path:
        return self._root

    @property
    def base_commit(self) -> str:
        return self._commit

    def state(self) -> GitRepositoryState:
        raw = self._run_bytes(
            self._root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        changed_paths = _parse_porcelain_paths(raw)
        return GitRepositoryState(
            repository_root=str(self._root),
            base_commit=self._commit,
            is_dirty=bool(changed_paths),
            changed_paths=changed_paths,
        )

    def list_blobs(self) -> tuple[RepositoryBlob, ...]:
        raw = self._run_bytes(
            self._root,
            "ls-tree",
            "-r",
            "-l",
            "-z",
            "--full-tree",
            self._commit,
        )
        blobs: list[RepositoryBlob] = []
        for record in raw.split(b"\0"):
            if not record:
                continue
            metadata, separator, path_bytes = record.partition(b"\t")
            if not separator:
                raise DomainError(ErrorCode.GIT_COMMAND_FAILED, "无法解析 git ls-tree 输出")
            parts = metadata.decode("ascii", errors="strict").split()
            if len(parts) != 4:
                raise DomainError(ErrorCode.GIT_COMMAND_FAILED, "git ls-tree 元数据字段异常")
            mode, object_type, object_id, size_text = parts
            if object_type != "blob":
                continue
            try:
                path = path_bytes.decode("utf-8", errors="strict")
                size_bytes = int(size_text)
            except (UnicodeDecodeError, ValueError) as exc:
                raise DomainError(
                    ErrorCode.GIT_COMMAND_FAILED,
                    "仓库包含无法安全解析的路径或对象大小",
                ) from exc
            blobs.append(
                RepositoryBlob(
                    path=path,
                    object_id=object_id,
                    mode=mode,
                    size_bytes=size_bytes,
                )
            )
        return tuple(sorted(blobs, key=lambda item: item.path))

    def read_bytes(self, path: str) -> bytes:
        path = validate_repository_path(path)
        cached = self._content_by_path.get(path)
        if cached is not None:
            return cached
        content = self._run_bytes(self._root, "cat-file", "blob", f"{self._commit}:{path}")
        self._content_by_path[path] = content
        return content

    def preload_blobs(self, blobs: tuple[RepositoryBlob, ...]) -> None:
        """Load regular blobs with one Git process and retain them for profile construction."""

        pending = [blob for blob in blobs if blob.path not in self._content_by_path]
        object_ids = tuple(dict.fromkeys(blob.object_id for blob in pending))
        if not object_ids:
            return
        request = "".join(f"{object_id}\n" for object_id in object_ids).encode("ascii")
        try:
            completed = subprocess.run(
                ("git", "-C", str(self._root), "cat-file", "--batch"),
                input=request,
                check=False,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DomainError(
                ErrorCode.GIT_COMMAND_FAILED,
                "无法批量读取 Git 对象",
            ) from exc
        if completed.returncode != 0:
            raise DomainError(ErrorCode.GIT_COMMAND_FAILED, "批量读取 Git 对象失败")
        content_by_object = _parse_batch_objects(completed.stdout, object_ids)
        for blob in pending:
            self._content_by_path[blob.path] = content_by_object[blob.object_id]

    @staticmethod
    def _run_text(root: Path, *args: str) -> str:
        return GitInspector._run_bytes(root, *args).decode("utf-8", errors="replace")

    @staticmethod
    def _run_bytes(root: Path, *args: str) -> bytes:
        try:
            completed = subprocess.run(
                ("git", "-C", str(root), *args),
                check=False,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DomainError(
                ErrorCode.GIT_COMMAND_FAILED,
                "无法执行只读 Git 命令",
                details={"operation": args[0] if args else "unknown"},
            ) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            code = (
                ErrorCode.NOT_GIT_REPOSITORY
                if args[:2] == ("rev-parse", "--show-toplevel")
                else ErrorCode.GIT_COMMAND_FAILED
            )
            raise DomainError(
                code,
                "Git 仓库读取失败",
                details={
                    "operation": args[0] if args else "unknown",
                    "stderr": stderr[:2_000],
                },
            )
        return completed.stdout


def _parse_porcelain_paths(raw: bytes) -> tuple[str, ...]:
    records = [record for record in raw.split(b"\0") if record]
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 4:
            raise DomainError(ErrorCode.GIT_COMMAND_FAILED, "无法解析 git status 输出")
        status = record[:2]
        paths.append(record[3:].decode("utf-8", errors="strict"))
        if status[:1] in {b"R", b"C"} or status[1:2] in {b"R", b"C"}:
            index += 1
            if index < len(records):
                paths.append(records[index].decode("utf-8", errors="strict"))
        index += 1
    return tuple(sorted(set(paths)))


def _is_hex_object_id(value: str) -> bool:
    return len(value) in range(40, 65) and all(
        character in "0123456789abcdef" for character in value
    )


def _parse_batch_objects(raw: bytes, object_ids: tuple[str, ...]) -> dict[str, bytes]:
    parsed: dict[str, bytes] = {}
    cursor = 0
    for requested_id in object_ids:
        header_end = raw.find(b"\n", cursor)
        if header_end < 0:
            raise DomainError(ErrorCode.GIT_COMMAND_FAILED, "git cat-file 批量输出缺少头部")
        header = raw[cursor:header_end].decode("ascii", errors="strict").split()
        if len(header) != 3 or header[0] != requested_id or header[1] != "blob":
            raise DomainError(ErrorCode.GIT_COMMAND_FAILED, "git cat-file 返回了非 blob 对象")
        try:
            size = int(header[2])
        except ValueError as exc:
            raise DomainError(ErrorCode.GIT_COMMAND_FAILED, "Git blob 大小无效") from exc
        content_start = header_end + 1
        content_end = content_start + size
        if content_end >= len(raw) or raw[content_end : content_end + 1] != b"\n":
            raise DomainError(ErrorCode.GIT_COMMAND_FAILED, "Git blob 内容长度不匹配")
        parsed[requested_id] = raw[content_start:content_end]
        cursor = content_end + 1
    return parsed
