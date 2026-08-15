$Utf8Encoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $Utf8Encoding
$OutputEncoding = $Utf8Encoding
$env:PYTHONUTF8 = "1"
$env:UV_NO_EDITABLE = "1"
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-NativeCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host "正在执行：$Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name 失败，退出码：$LASTEXITCODE"
    }
}

Push-Location $ProjectRoot
try {
    Invoke-NativeCheck "Ruff 静态检查" { uv run ruff check . }
    Invoke-NativeCheck "Ruff 格式检查" { uv run ruff format --check . }
    Invoke-NativeCheck "mypy 类型检查" { uv run mypy src }
    Invoke-NativeCheck "pytest 离线测试" { uv run pytest }
    Invoke-NativeCheck "Python 包导入" { uv run python -c "import evoweave_ds; print(evoweave_ds.__version__)" }
    & (Join-Path $PSScriptRoot "检查项目结构.ps1")
}
finally {
    Pop-Location
}

Write-Host "EvoWeave 全部阶段门禁通过。"
