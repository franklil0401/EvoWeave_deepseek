$Utf8Encoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $Utf8Encoding
$OutputEncoding = $Utf8Encoding
$env:PYTHONUTF8 = "1"
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

$RequiredPaths = @(
    "任务文档.md",
    "项目结构文档.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".pre-commit-config.yaml",
    ".github/workflows/ci.yml",
    "src/evoweave_ds/domain/ports.py",
    "src/evoweave_ds/infrastructure/models/fake.py",
    "tests/contract"
)

$MissingPaths = @(
    foreach ($RelativePath in $RequiredPaths) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $RelativePath))) {
            $RelativePath
        }
    }
)
if ($MissingPaths.Count -gt 0) {
    throw "缺少项目结构要求的路径：$($MissingPaths -join ', ')"
}

$IgnoredDirectoryNames = @(".git", ".venv", ".runtime", ".pytest_cache", ".mypy_cache", ".ruff_cache")
$MarkdownFiles = Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -Filter "*.md" | Where-Object {
    $FullName = $_.FullName
    -not ($IgnoredDirectoryNames | Where-Object { $FullName -like "*\$_\*" })
}
$InvalidMarkdownNames = @(
    foreach ($File in $MarkdownFiles) {
        if ($File.Name -ne "README.md" -and $File.BaseName -notmatch "[\p{IsCJKUnifiedIdeographs}]") {
            $File.FullName.Substring($ProjectRoot.Length + 1)
        }
    }
)
if ($InvalidMarkdownNames.Count -gt 0) {
    throw "除 README.md 外，Markdown 文件名必须包含中文：$($InvalidMarkdownNames -join ', ')"
}

$ForbiddenRoleFiles = @("planner_agent.py", "developer_agent.py", "reviewer_agent.py", "tester_agent.py")
$DetectedRoleFiles = @(
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "src") -Recurse -File | Where-Object {
        $_.Name -in $ForbiddenRoleFiles
    }
)
if ($DetectedRoleFiles.Count -gt 0) {
    throw "检测到固定业务角色文件：$($DetectedRoleFiles.FullName -join ', ')"
}

$GitIgnore = Get-Content -Raw -LiteralPath (Join-Path $ProjectRoot ".gitignore")
if ($GitIgnore -notmatch "(?m)^\.runtime/$") {
    throw ".gitignore 必须忽略 .runtime/"
}

$DomainFiles = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "src/evoweave_ds/domain") -File -Filter "*.py"
$ForbiddenDomainImports = $DomainFiles | Select-String -Pattern "^from evoweave_ds\.(?!domain)|^import evoweave_ds\.(?!domain)"
if ($ForbiddenDomainImports) {
    throw "domain 层不能依赖其他项目层：$($ForbiddenDomainImports.Path -join ', ')"
}

$StructureDocument = Get-Item -LiteralPath (Join-Path $ProjectRoot "项目结构文档.md")
$StructureDocumentContent = Get-Content -Raw -LiteralPath $StructureDocument.FullName
$DocumentDateMatch = [regex]::Match($StructureDocumentContent, "最近更新：(\d{4}-\d{2}-\d{2})")
if (-not $DocumentDateMatch.Success) {
    throw "项目结构文档.md 必须记录最近更新日期"
}
$DocumentDate = [datetime]::ParseExact(
    $DocumentDateMatch.Groups[1].Value,
    "yyyy-MM-dd",
    [Globalization.CultureInfo]::InvariantCulture
)
$TrackedStructureRoots = @("src", "tests", ".github", "scripts") | ForEach-Object {
    Get-Item -LiteralPath (Join-Path $ProjectRoot $_)
}
$TrackedDirectories = $TrackedStructureRoots + @(
    Get-ChildItem -LiteralPath $TrackedStructureRoots.FullName -Recurse -Directory | Where-Object {
        $_.FullName -notmatch "\\(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)\\?"
    }
)
$TrackedConfiguration = @("pyproject.toml", "uv.lock", ".pre-commit-config.yaml") | ForEach-Object {
    Get-Item -LiteralPath (Join-Path $ProjectRoot $_)
}
$LatestStructureChange = $TrackedDirectories + $TrackedConfiguration |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($LatestStructureChange -and $DocumentDate.Date -lt $LatestStructureChange.LastWriteTime.Date) {
    throw "项目结构文档.md 早于最近一次结构变更，请先同步活文档"
}

Write-Host "项目结构检查通过。"
