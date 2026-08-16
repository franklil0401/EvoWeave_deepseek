"""Hidden acceptance checks for the EvoWeave_deepseek Flask benchmark.

Runs inside the candidate worktree (cwd = repository root) and asserts the
modified sources satisfy each Flask enhancement requirement via static
source assertions (baseline must fail, correct implementation passes).
"""

import pathlib
import re
import sys

ROOT = pathlib.Path.cwd()


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="replace")


def check_flask_01_static_cache_extension() -> None:
    """get_send_file_max_age 按扩展名提供默认缓存时间。"""
    app = _read("src/flask/app.py")
    assert "get_send_file_max_age" in app, "缺少 get_send_file_max_age"
    assert re.search(r"EXTENSION|suffix|split|\.png|\.css|casefold", app), "未按扩展名处理"
    assert "3600" in app, "缺少默认缓存时间 3600"
    assert re.search(r"def get_send_file_max_age", app), "缺少方法定义"


def check_flask_02_json_sort_config() -> None:
    """DefaultJSONProvider.dumps 读取 JSON_SORT_KEYS 配置。"""
    provider = _read("src/flask/json/provider.py")
    assert "JSON_SORT_KEYS" in provider, "未读取 JSON_SORT_KEYS 配置"
    assert "sort_keys" in provider, "缺少 sort_keys 逻辑"
    assert re.search(r"config\.get\(['\"]JSON_SORT_KEYS", provider), "未从 app.config 读取"


def check_flask_03_envvar_required() -> None:
    """Config.from_envvar 支持 required 参数。"""
    config = _read("src/flask/config.py")
    assert "required" in config, "from_envvar 缺少 required 参数"
    assert re.search(r"def from_envvar\(.*required", config), "required 参数缺失"
    assert re.search(r"required.*(?:raise|RuntimeError|ValueError)|if required", config), "required=True 时缺少明确报错"


def check_flask_04_namespace_lower_option() -> None:
    """Config.get_namespace 支持 lower 参数。"""
    config = _read("src/flask/config.py")
    assert "lower" in config, "get_namespace 缺少 lower 参数"
    assert re.search(r"def get_namespace\(.*lower", config), "lower 参数缺失"


def check_flask_05_client_redirect_config() -> None:
    """FlaskClient.open 支持 TESTING_FOLLOW_REDIRECTS 配置。"""
    testing = _read("src/flask/testing.py")
    app_src = _read("src/flask/app.py")
    assert "TESTING_FOLLOW_REDIRECTS" in testing or "TESTING_FOLLOW_REDIRECTS" in app_src, "缺少 TESTING_FOLLOW_REDIRECTS 配置读取"
    assert "follow_redirects" in testing, "open 缺少 follow_redirects 处理"


def check_flask_06_methods_validate_option() -> None:
    """add_url_rule 支持 validate_methods 可选校验开关。"""
    app = _read("src/flask/app.py")
    assert "validate_methods" in app, "add_url_rule 缺少 validate_methods 参数"
    assert re.search(r"def add_url_rule\(.*validate_methods", app), "validate_methods 参数缺失"
    assert re.search(r"validate_methods.*(?:ValueError|raise)", app), "校验失败时缺少 ValueError"


def check_flask_07_static_security_headers() -> None:
    """send_static_file 附加 X-Content-Type-Options 安全头。"""
    app = _read("src/flask/app.py")
    assert "X-Content-Type-Options" in app, "缺少 X-Content-Type-Options 头"
    assert "nosniff" in app, "缺少 nosniff 值"
    assert "SEND_STATIC_SECURITY_HEADERS" in app, "缺少配置开关"


def check_flask_08_errorhandler_dispatch() -> None:
    """make_http_error_response 可覆盖钩子。"""
    app = _read("src/flask/app.py")
    assert "make_http_error_response" in app, "缺少 make_http_error_response 钩子"
    assert re.search(r"def make_http_error_response", app), "钩子方法未定义"
    assert "handle_http_exception" in app, "缺少 handle_http_exception"


_CHECKS = {
    "bench-11-static-cache-extension": check_flask_01_static_cache_extension,
    "bench-12-json-sort-config": check_flask_02_json_sort_config,
    "bench-13-envvar-required": check_flask_03_envvar_required,
    "bench-14-namespace-lower-option": check_flask_04_namespace_lower_option,
    "bench-15-client-redirect-config": check_flask_05_client_redirect_config,
    "bench-16-methods-validate-option": check_flask_06_methods_validate_option,
    "bench-17-static-security-headers": check_flask_07_static_security_headers,
    "bench-18-errorhandler-dispatch": check_flask_08_errorhandler_dispatch,
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: hidden_acceptance_flask.py <benchmark_id>")
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