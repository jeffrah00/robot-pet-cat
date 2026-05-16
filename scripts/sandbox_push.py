#!/usr/bin/env python3
"""Push specific files to GitHub from the sandbox via the REST API.

This exists because:
  - The Linux sandbox can't run `git` against the Windows-mounted workspace
    folder (corrupts .git/).
  - Driving PowerShell via computer-use is blocked at tier-click.
  - There's no GitHub MCP connector in the registry.

So we push via REST. The git data API lets us create one commit containing
multiple file changes in a single atomic operation.

Setup:
  1. Generate a Personal Access Token at:
       https://github.com/settings/tokens/new
     Scope: `repo` (private) or `public_repo` (public). Pick a sane expiry.
  2. Add it to .env:  GITHUB_TOKEN=ghp_xxxxxxxxxxxx
  3. (Optional) override REPO / BRANCH via env or CLI flags.

Usage:
  python scripts/sandbox_push.py -m "Phase 2: AMP trainer scaffold" \
      src/robot_pet_cat/motion/amp_trainer.py \
      configs/motion/cat_amp.yaml

  python scripts/sandbox_push.py -m "Fix typo" docs/retargeting.md
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "jeffrah00/robot-pet-cat"
DEFAULT_BRANCH = "main"
API = "https://api.github.com"


def load_env_file(path: Path) -> dict:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def get_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    file_env = load_env_file(Path(".env"))
    if file_env.get("GITHUB_TOKEN"):
        return file_env["GITHUB_TOKEN"]
    print(
        "ERROR: GITHUB_TOKEN not found.\n"
        "  1. Create a PAT at https://github.com/settings/tokens/new (scope: repo)\n"
        "  2. Add GITHUB_TOKEN=... to .env (or pass --token).",
        file=sys.stderr,
    )
    sys.exit(2)


def gh(
    method: str,
    path: str,
    token: str,
    body: dict | None = None,
) -> dict:
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "robot-pet-cat-sandbox-push/0.1")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        print(f"GitHub API {method} {path} -> HTTP {e.code}\n{msg}", file=sys.stderr)
        raise


def repo_relative(path: Path, repo_root: Path) -> str:
    """Compute path relative to the project root, with forward slashes."""
    return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")


def push(
    files: list[Path],
    message: str,
    repo: str,
    branch: str,
    token: str,
    repo_root: Path,
    dry_run: bool = False,
) -> str:
    if dry_run:
        head_sha = "DRYRUN-HEAD"
        base_tree_sha = "DRYRUN-TREE"
        print(f"[push] DRY RUN against {repo}@{branch}")
    else:
        # 1. Get current HEAD ref + commit + tree.
        ref = gh("GET", f"/repos/{repo}/git/ref/heads/{branch}", token)
        head_sha = ref["object"]["sha"]
        head_commit = gh("GET", f"/repos/{repo}/git/commits/{head_sha}", token)
        base_tree_sha = head_commit["tree"]["sha"]
        print(f"[push] base: {repo}@{branch} {head_sha[:8]}")

    # 2. For each file, create a blob (or note deletion).
    tree_entries: list[dict] = []
    for f in files:
        rel = repo_relative(f, repo_root)
        if not f.exists():
            tree_entries.append({
                "path": rel, "mode": "100644", "type": "blob", "sha": None,
            })
            print(f"[push] D  {rel}")
            continue
        content = f.read_bytes()
        if dry_run:
            tree_entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": "DRYRUN"})
            print(f"[push] M  {rel}  ({len(content)} bytes)")
            continue
        blob = gh("POST", f"/repos/{repo}/git/blobs", token, body={
            "content": base64.b64encode(content).decode("ascii"),
            "encoding": "base64",
        })
        tree_entries.append({
            "path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"],
        })
        print(f"[push] M  {rel}  ({len(content)} bytes)  blob {blob['sha'][:8]}")

    if dry_run:
        print(f"[push] DRY RUN: would commit {len(tree_entries)} files: {message!r}")
        return "DRYRUN"

    tree = gh("POST", f"/repos/{repo}/git/trees", token, body={
        "base_tree": base_tree_sha,
        "tree": tree_entries,
    })
    commit = gh("POST", f"/repos/{repo}/git/commits", token, body={
        "message": message,
        "tree": tree["sha"],
        "parents": [head_sha],
    })
    gh("PATCH", f"/repos/{repo}/git/refs/heads/{branch}", token, body={
        "sha": commit["sha"],
    })

    print(f"[push] committed {commit['sha'][:8]} -> {branch}")
    print(f"[push] view at https://github.com/{repo}/commit/{commit['sha']}")
    return commit["sha"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path,
                   help="Files to push (paths relative to repo root or absolute).")
    p.add_argument("-m", "--message", required=True, help="Commit message.")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPO", DEFAULT_REPO),
                   help=f"owner/name (default: {DEFAULT_REPO}).")
    p.add_argument("--branch", default=os.environ.get("GITHUB_BRANCH", DEFAULT_BRANCH),
                   help=f"target branch (default: {DEFAULT_BRANCH}).")
    p.add_argument("--token", default=None, help="Override GITHUB_TOKEN from env/.env.")
    p.add_argument("--repo-root", type=Path, default=Path.cwd(),
                   help="Repo root for computing relative paths (default: cwd).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be pushed without calling the API.")
    args = p.parse_args()

    token = get_token(args.token)
    push(
        files=args.files,
        message=args.message,
        repo=args.repo,
        branch=args.branch,
        token=token,
        repo_root=args.repo_root,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
