# scripts/sync.ps1 -- stage, commit, and push all changes in the project.
# ASCII only (PowerShell on Windows reads .ps1 as Windows-1252 unless BOM'd).
#
# Usage:
#   .\scripts\sync.ps1                         # one-shot: add . + commit + push
#   .\scripts\sync.ps1 -m "your message"       # one-shot with explicit message
#   .\scripts\sync.ps1 -Watch                  # poll every 60s and sync if dirty
#   .\scripts\sync.ps1 -Watch -IntervalSec 30  # poll every 30s instead

[CmdletBinding()]
param(
    [Alias("m")]
    [string]$Message = "",
    [switch]$Watch,
    [int]$IntervalSec = 60
)

if (-not (Test-Path ".\pyproject.toml")) {
    Write-Error "Run this from the project root (folder with pyproject.toml)."
}

function Get-AutoMessage {
    $changes = git status --porcelain
    $files = $changes | ForEach-Object { ($_ -replace '^...', '').Trim() }
    $count = ($files | Measure-Object).Count
    $sample = ($files | Select-Object -First 3) -join ", "
    if ($count -le 3) { return "sync: $sample" }
    return "sync: $sample, +$($count - 3) more"
}

function Sync-Once {
    param([string]$Msg = "")
    $changes = git status --porcelain
    if (-not $changes) {
        Write-Host "[sync] nothing to commit"
        return
    }
    git add . | Out-Null
    if (-not $Msg) { $Msg = Get-AutoMessage }
    git commit -m $Msg | Out-Null
    Write-Host "[sync] committed: $Msg"
    git push 2>&1 | ForEach-Object { Write-Host "        $_" }
    Write-Host "[sync] pushed"
}

if ($Watch) {
    Write-Host "[sync] watching every $IntervalSec seconds. Ctrl-C to stop."
    while ($true) {
        try { Sync-Once } catch { Write-Warning $_ }
        Start-Sleep -Seconds $IntervalSec
    }
} else {
    Sync-Once -Msg $Message
}
