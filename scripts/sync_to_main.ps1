# Sync the local working tree to origin/main.
# Sandbox can't safely run git ops on a Windows-mounted folder; this PS1
# helper does it from the host side.
#
# Usage (PowerShell):
#   cd "C:\Users\jeffr\Documents\Claude\Projects\Robot pet cat"
#   powershell -ExecutionPolicy Bypass -File scripts\sync_to_main.ps1
#
# What it does:
#   1. git fetch origin
#   2. show what would change (stash any local edits if present)
#   3. hard-reset main to origin/main
#   4. clean untracked render logs / scratch files (KEEPS .env, .venv, models/, checkpoints/, renders/)
#
# Safe to re-run. Will NOT delete your weights, renders, .env, or virtualenv.

$ErrorActionPreference = "Stop"

Write-Host "=== git status before sync ==="
git status -s

Write-Host ""
Write-Host "=== fetching origin ==="
git fetch origin

Write-Host ""
Write-Host "=== local vs origin/main ==="
$ahead  = git rev-list --count origin/main..HEAD
$behind = git rev-list --count HEAD..origin/main
Write-Host "  ahead $ahead , behind $behind"

if ((git status --porcelain | Measure-Object).Count -gt 0) {
    Write-Host ""
    Write-Host "=== stashing local changes (safety net) ==="
    git stash push -u -m "auto-stash before sync_to_main.ps1"
    Write-Host "  -> recover with 'git stash list' / 'git stash pop' if needed"
}

Write-Host ""
Write-Host "=== hard-reset main to origin/main ==="
git checkout main
git reset --hard origin/main

Write-Host ""
Write-Host "=== cleaning untracked scratch files (keeping .env, .venv, models/, checkpoints/, renders/) ==="
git clean -fd `
    -e ".env" `
    -e ".venv" `
    -e "models" `
    -e "checkpoints" `
    -e "renders" `
    -e "wandb" `
    -e "data"

Write-Host ""
Write-Host "=== final state ==="
git log --oneline -5
Write-Host ""
git status -s
Write-Host ""
Write-Host "Done. Local working tree now matches origin/main."
