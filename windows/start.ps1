# Start mnemerynne-balancer on Windows (development): NVIDIA cards, llama.cpp's CUDA build.
#
#   .\windows\start.ps1                     sync, then serve in this window (Ctrl+C stops it and its llama-servers)
#   .\windows\start.ps1 -Devices            just list the cards llama-server sees
#   .\windows\start.ps1 -LlamaServer D:\llama.cpp\llama-server.exe
#   .\windows\start.ps1 -NoSync             skip "uv sync" (offline, or when nothing changed)
#
# The first run makes .env from .env.example with Windows values: llama-server from C:\llama.cpp,
# data in .\data, and a new control-page password and API key (printed once). An existing .env keeps
# its values; only the Linux defaults and empty password/key are replaced.
[CmdletBinding()]
param(
    [string]$LlamaServer = "C:\llama.cpp\llama-server.exe",
    [switch]$Devices,
    [switch]$NoSync
)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$EnvFile = Join-Path $Root ".env"

# 1. Python environment: the exact versions in uv.lock.
if (-not $NoSync -or -not (Test-Path $Python)) {
    py -m uv --version *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing uv for this Python (py -m pip install uv)..."
        py -m pip install --quiet --user uv
        if ($LASTEXITCODE -ne 0) { throw "Couldn't install uv. Install Python 3.12+ from python.org (with the py launcher)." }
    }
    py -m uv sync --frozen --no-dev
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed (see above)." }
}

# 2. .env with Windows values.
function Set-EnvValue([string[]]$Lines, [string]$Key, [string]$Value) {
    $found = $false
    $out = foreach ($line in $Lines) {
        if ($line -match "^$Key=") { $found = $true; "$Key=$Value" } else { $line }
    }
    if (-not $found) { $out = @($out) + "$Key=$Value" }
    return ,@($out)
}
function Get-EnvValue([string[]]$Lines, [string]$Key) {
    $match = $Lines | Where-Object { $_ -match "^$Key=" } | Select-Object -Last 1
    if ($match) { return $match.Substring($Key.Length + 1).Trim() } else { return "" }
}
function New-Secret([int]$Bytes) {
    & $Python -c "import secrets; print(secrets.token_urlsafe($Bytes))"
}

if (-not (Test-Path $EnvFile)) { Copy-Item (Join-Path $Root ".env.example") $EnvFile }
$lines = @(Get-Content $EnvFile)
$newPassword = $newKey = $null

$llama = Get-EnvValue $lines "LLAMA_SERVER"
if (-not $llama -or $llama.StartsWith("/") -or $PSBoundParameters.ContainsKey("LlamaServer")) {
    $llama = $LlamaServer
    $lines = Set-EnvValue $lines "LLAMA_SERVER" $llama
}
if ((Get-EnvValue $lines "BALANCER_DATA_DIR").StartsWith("/")) {
    $lines = Set-EnvValue $lines "BALANCER_DATA_DIR" ""          # empty: .\data next to the code
}
if ((Get-EnvValue $lines "ADMIN_PASSWORD").Length -lt 8) {
    $newPassword = New-Secret 15
    $lines = Set-EnvValue $lines "ADMIN_PASSWORD" $newPassword
}
if ((Get-EnvValue $lines "BALANCER_API_KEY").Length -lt 16) {
    $newKey = New-Secret 32
    $lines = Set-EnvValue $lines "BALANCER_API_KEY" $newKey
}
Set-Content -Path $EnvFile -Value $lines -Encoding utf8

# 3. llama-server.
if (-not (Test-Path $llama)) {
    Write-Host ""
    Write-Host "!! llama-server isn't at $llama." -ForegroundColor Red
    Write-Host "   Download a win-cuda-12.x-x64 release and its cudart zip from"
    Write-Host "   https://github.com/ggml-org/llama.cpp/releases, unzip both into C:\llama.cpp,"
    Write-Host "   or run this with -LlamaServer <path to llama-server.exe>."
    exit 1
}

if ($newPassword -or $newKey) {
    Write-Host ""
    Write-Host "Saved in $EnvFile (shown once):" -ForegroundColor Yellow
    if ($newPassword) { Write-Host "  Control page password: $newPassword" }
    if ($newKey) { Write-Host "  API key (Mnemerynne's Settings -> Balancer nodes): $newKey" }
}

# 4. Run.
if ($Devices) {
    & $Python -m mnemerynne_balancer devices
    exit $LASTEXITCODE
}
$port = Get-EnvValue $lines "BALANCER_PORT"
if (-not $port) { $port = "9180" }
Write-Host ""
Write-Host "mnemerynne-balancer: http://localhost:$port  (Ctrl+C stops it and its llama-servers)" -ForegroundColor Green
& $Python -m mnemerynne_balancer serve
exit $LASTEXITCODE
