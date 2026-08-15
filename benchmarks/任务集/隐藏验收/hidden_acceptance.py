"""Hidden acceptance checks for EvoWeave_deepseek real-repository benchmark.

Runs inside the candidate worktree (cwd = repository root) and asserts the
modified sources satisfy each requirement without importing third-party deps.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path.cwd()


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="replace")


def check_bench_01_short_merge_threshold() -> None:
    """合并阈值/强制入库超时参数化（TA_MIN_CHARS/TA_FLUSH_SECONDS）。"""
    listener = _read("app/audio_listener.py")
    config = _read("app/config.py")
    assert "merge_min_chars" in listener, "audio_listener 未使用合并阈值配置"
    assert "TA_MIN_CHARS" in listener or "TA_MIN_CHARS" in config, "缺少 TA_MIN_CHARS 环境变量读取"
    assert re.search(r"_MERGE_MAX_CHARS\s*=\s*500", listener) is None, "仍存在硬编码 500 阈值"
    assert "TA_FLUSH_SECONDS" in listener or "TA_FLUSH_SECONDS" in config, "缺少 TA_FLUSH_SECONDS"


def check_bench_02_corrupt_transcript() -> None:
    """课堂记录文件损坏容错。"""
    store = _read("rag/transcript_store.py")
    assert "continue" in store, "iter_records 缺少跳过逻辑"
    assert re.search(r"json\.loads", store), "iter_records 缺少解析逻辑"
    assert "except" in store, "iter_records 缺少异常处理"


def check_bench_03_chunk_defaults() -> None:
    """chunk_text 默认切分参数与 Settings 一致（800/120）。"""
    chunking = _read("rag/chunking.py")
    assert re.search(r"def chunk_text\(text: str, max_chars: int = 800", chunking), "默认 max_chars 不是 800"
    assert re.search(r"overlap: int = 120", chunking), "默认 overlap 不是 120"


def check_bench_04_clear_chat() -> None:
    """Web UI 单独清空问答对话，不影响课堂记录。"""
    server = _read("app/web_server.py")
    pipeline = _read("rag/pipeline.py")
    assert "clear_chat" in server or "clear-chat" in server, "web_server 缺少清空对话接口"
    assert "clear_student" in pipeline or "clear_student_questions" in pipeline, "pipeline 缺少按来源清理能力"


def check_bench_05_transcribe_prompt() -> None:
    """豆包转写提示词可配置（TA_DOUBAO_TRANSCRIBE_PROMPT）。"""
    config = _read("app/config.py")
    transcriber = _read("app/transcriber.py")
    assert "TA_DOUBAO_TRANSCRIBE_PROMPT" in config, "config 缺少 TA_DOUBAO_TRANSCRIBE_PROMPT"
    assert "transcribe_prompt" in config, "Settings 缺少转写提示词字段"
    assert "prompt" in transcriber, "transcriber 未使用提示词配置"


def check_bench_06_atomic_rebuild() -> None:
    """索引重建原子化。"""
    index = _read("rag/vector_index.py")
    pipeline = _read("rag/pipeline.py")
    assert "replace" in index or "tmp" in index or "temporary" in index, "vector_index 缺少原子替换"
    assert "rebuild" in pipeline, "pipeline 缺少 rebuild 入口"


def check_bench_07_transcribe_retry() -> None:
    """转写失败自动重试一次。"""
    transcriber = _read("app/transcriber.py")
    assert "retry" in transcriber, "transcriber 缺少重试逻辑"
    assert re.search(r"for .* in range\(2\)|attempt|retry", transcriber), "未发现重试机制"


def check_bench_08_record_source_filter() -> None:
    """最近记录按来源过滤。"""
    store = _read("rag/transcript_store.py")
    pipeline = _read("rag/pipeline.py")
    assert re.search(r"def recent\(self, limit: int = 5, source: str \| None = None\)", store), "recent 缺少 source 参数"
    assert "recent_records" in pipeline, "pipeline 缺少 recent_records"


_CHECKS = {
    "bench-01-short-merge-threshold": check_bench_01_short_merge_threshold,
    "bench-02-corrupt-transcript": check_bench_02_corrupt_transcript,
    "bench-03-chunk-defaults": check_bench_03_chunk_defaults,
    "bench-04-clear-chat": check_bench_04_clear_chat,
    "bench-05-transcribe-prompt": check_bench_05_transcribe_prompt,
    "bench-06-atomic-rebuild": check_bench_06_atomic_rebuild,
    "bench-07-transcribe-retry": check_bench_07_transcribe_retry,
    "bench-08-record-source-filter": check_bench_08_record_source_filter,
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: hidden_acceptance.py <benchmark_id>")
        return 2
    benchmark_id = sys.argv[1]
    check = _CHECKS.get(benchmark_id)
    if check is None:
        print(f"unknown benchmark_id: {benchmark_id}")
        return 2
    try:
        check()
    except AssertionError as exc:
        print(f"FAIL {benchmark_id}: {exc}")
        return 1
    print(f"PASS {benchmark_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
