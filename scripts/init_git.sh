#!/usr/bin/env bash
# WSL / Linux / macOS equivalent of init_git.ps1. Run from the project root:
#   bash scripts/init_git.sh
set -euo pipefail

USER_NAME="${USER_NAME:-jeffrah00}"
USER_EMAIL="${USER_EMAIL:-jeffrah89@gmail.com}"
REPO_NAME="${REPO_NAME:-robot-pet-cat}"
GITHUB_USER="${GITHUB_USER:-jeffrah00}"

if [[ ! -f pyproject.toml ]]; then
  echo "Run this from the project root (the folder containing pyproject.toml)."
  exit 1
fi

# Clean up sandbox leftover
rm -rf .git.broken || true

if [[ ! -d .git ]]; then
  git init -b main
fi

git config user.name  "$USER_NAME"
git config user.email "$USER_EMAIL"

if [[ -n "$(git status --porcelain)" ]]; then
  git add .
  git commit -m "Initial scaffold: docs, configs, src skeleton, sim smoke test"
fi

cat <<EOF

==================== Next steps ====================
1. Create an EMPTY repo on GitHub at: https://github.com/new
   Name it '$REPO_NAME'. Don't add README/.gitignore/license.

2. Add the remote and push:

   # SSH:
   git remote add origin git@github.com:$GITHUB_USER/$REPO_NAME.git
   git push -u origin main

   # …or HTTPS:
   git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git
   git push -u origin main
====================================================
EOF
