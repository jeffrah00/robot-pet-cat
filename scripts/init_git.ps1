# Initialize git for the robot-pet-cat repo. Run from PowerShell in the project root:
#   cd "C:\Users\jeffr\Documents\Claude\Projects\Robot pet cat"
#   .\scripts\init_git.ps1
#
# What it does:
#   1. Removes the .git.broken leftover from sandbox init (if present)
#   2. Runs git init on the main branch
#   3. Configures your name + email
#   4. Stages all files and makes the initial commit
#   5. Prints the exact commands to push to a GitHub repo

[CmdletBinding()]
param(
    [string]$UserName  = "jeffrah00",
    [string]$UserEmail = "jeffrah89@gmail.com",
    [string]$RepoName  = "robot-pet-cat"
)

$ErrorActionPreference = "Stop"

# Sanity check: are we in the project root?
if (-not (Test-Path ".\pyproject.toml")) {
    Write-Error "Run this from the project root (the folder containing pyproject.toml)."
}

# 1. clean up sandbox leftover
if (Test-Path ".\.git.broken") {
    Write-Host ">>> Removing leftover .git.broken from sandbox init"
    Remove-Item -Recurse -Force ".\.git.broken"
}
if (Test-Path ".\.git") {
    Write-Host ">>> .git already exists — skipping init"
} else {
    Write-Host ">>> git init"
    git init -b main | Out-Null
}

# 2. local user config
git config user.name  $UserName
git config user.email $UserEmail

# 3. first commit (idempotent)
$staged = git status --porcelain
if ($staged) {
    Write-Host ">>> git add ."
    git add .
    Write-Host ">>> initial commit"
    git commit -m "Initial scaffold: docs, configs, src skeleton, sim smoke test" | Out-Null
} else {
    Write-Host ">>> nothing to commit (already up to date)"
}

Write-Host ""
Write-Host "==================== Next steps ===================="
Write-Host "1. Create an EMPTY repo on GitHub at:"
Write-Host "     https://github.com/new"
Write-Host "   Name it '$RepoName'. Don't add README/.gitignore/license — the scaffold has them."
Write-Host ""
Write-Host "2. Add the remote and push. Replace <your-github-username>:"
Write-Host ""
Write-Host "   # SSH (recommended if you have an SSH key set up):"
Write-Host "   git remote add origin git@github.com:<your-github-username>/$RepoName.git"
Write-Host "   git push -u origin main"
Write-Host ""
Write-Host "   # …or HTTPS:"
Write-Host "   git remote add origin https://github.com/<your-github-username>/$RepoName.git"
Write-Host "   git push -u origin main"
Write-Host "===================================================="
