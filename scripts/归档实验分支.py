"""Archive experiment branch info from the web_learning_tool clone.

After the full matrix finishes, this script snapshots every evoweave-exp/* and
evoweave_ds/run_* branch (name, HEAD sha, base sha) into
benchmarks/结果/实验分支归档/ as JSON + markdown for manual review.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(r"E:\mutil_Agent\deepseek\exp-repos\web_learning_tool")
OUTPUT = Path(__file__).resolve().parent.parent / "benchmarks" / "结果" / "实验分支归档"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def main() -> None:
    branches = _git(
        "for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads"
    ).splitlines()
    records = []
    for line in branches:
        name, head = line.split(" ", 1)
        if not (name.startswith("evoweave-exp/") or name.startswith("evoweave_ds/")):
            continue
        base = _git("merge-base", "main", name)
        records.append({"branch": name, "head_sha": head, "base_sha": base})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {"repository": str(REPO), "branches": records}
    (OUTPUT / "实验分支归档.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 实验分支归档",
        "",
        f"- 仓库：`{REPO}`",
        f"- 分支数：{len(records)}",
        "",
        "| 分支 | HEAD | 基线 |",
        "|---|---|---|",
    ]
    lines += [f"| {r['branch']} | `{r['head_sha']}` | `{r['base_sha']}` |" for r in records]
    (OUTPUT / "实验分支归档.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"archived {len(records)} branches -> {OUTPUT}")


if __name__ == "__main__":
    main()
