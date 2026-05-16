#!/usr/bin/env python3
"""Download one Pexels video into data/motion_clips_raw/ via the Pexels API.

Why the Pexels API and not yt-dlp:
  Pexels puts its public pages behind Cloudflare's anti-bot challenge, so the
  yt-dlp `generic` extractor returns HTTP 403. The Pexels API is the sanctioned
  path and returns clean direct .mp4 URLs.

Requirements:
  - Free Pexels API key: https://www.pexels.com/api/new/
  - Put it in .env as PEXELS_API_KEY=... (or pass via --api-key)

Usage:
    python scripts/fetch_clip.py 34482817
    python scripts/fetch_clip.py https://www.pexels.com/video/gray-cat-walking-outdoors-on-a-cloudy-day-34482817/
    python scripts/fetch_clip.py 34482817 --max-height 1080
    python scripts/fetch_clip.py 34482817 --then-extract
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PEXELS_VIDEO_API = "https://api.pexels.com/videos/videos/{video_id}"


def parse_video_id(arg: str) -> str:
    """Accept a numeric id, a full Pexels URL, or a slug-with-id URL."""
    if arg.isdigit():
        return arg
    m = re.search(r"/video/(?:[^/]+?-)?(\d+)/?", arg)
    if m:
        return m.group(1)
    raise ValueError(f"Couldn't extract a video ID from {arg!r}")


def load_env_file(path: Path) -> dict:
    """Minimal .env loader so we don't require python-dotenv at script time."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def get_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    env_key = os.environ.get("PEXELS_API_KEY")
    if env_key:
        return env_key
    file_env = load_env_file(Path(".env"))
    if file_env.get("PEXELS_API_KEY"):
        return file_env["PEXELS_API_KEY"]
    print(
        "ERROR: PEXELS_API_KEY not found.\n"
        "  1. Get a free key at https://www.pexels.com/api/new/\n"
        "  2. Add PEXELS_API_KEY=... to .env (or pass --api-key)",
        file=sys.stderr,
    )
    sys.exit(2)


def fetch_video_meta(video_id: str, api_key: str) -> dict:
    url = PEXELS_VIDEO_API.format(video_id=video_id)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": api_key,
            "User-Agent": "robot-pet-cat/0.0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Pexels API returned HTTP {e.code}: {body}") from e


def pick_mp4(meta: dict, max_height: int) -> dict:
    """From the video_files list, pick the highest-resolution mp4 <= max_height.

    Fall back to the highest available mp4 if none satisfy max_height.
    """
    files = [
        f for f in meta.get("video_files", [])
        if f.get("file_type") == "video/mp4"
    ]
    if not files:
        raise SystemExit(f"No mp4 files in video metadata: {meta}")

    capped = [f for f in files if (f.get("height") or 0) <= max_height]
    pool = capped or files
    return max(pool, key=lambda f: (f.get("height") or 0, f.get("width") or 0))


def slug_for_meta(meta: dict, video_id: str) -> str:
    page_url = meta.get("url", "")
    m = re.search(r"/video/([^/]+?)-(\d+)/?", page_url)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return f"pexels-{video_id}"


def download_to(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "robot-pet-cat/0.0.1"},
    )
    with urllib.request.urlopen(req, timeout=120) as r, open(out_path, "wb") as f:
        total = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            total += len(chunk)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="Pexels URL or numeric video id.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output mp4 path (default: data/motion_clips_raw/<slug>.mp4).")
    p.add_argument("--max-height", type=int, default=1080,
                   help="Cap on video height in pixels (default 1080; 4K is overkill for pose).")
    p.add_argument("--api-key", default=None,
                   help="Pexels API key (overrides PEXELS_API_KEY env / .env).")
    p.add_argument("--then-extract", action="store_true",
                   help="After download, run scripts/extract_keypoints.py on the file.")
    args = p.parse_args()

    api_key = get_api_key(args.api_key)
    video_id = parse_video_id(args.source)

    print(f"[fetch] querying Pexels API for video {video_id}")
    meta = fetch_video_meta(video_id, api_key)

    chosen = pick_mp4(meta, args.max_height)
    print(
        f"[fetch] picked {chosen.get('width')}x{chosen.get('height')} "
        f"@ {chosen.get('fps')}fps  ({chosen.get('quality')})"
    )

    out = args.out or Path("data/motion_clips_raw") / f"{slug_for_meta(meta, video_id)}.mp4"
    print(f"[fetch] downloading -> {out}")
    download_to(chosen["link"], out)
    print(f"[fetch] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    if args.then_extract:
        # Pose extraction lives in the dedicated .venv-pose (separate from
        # .venv to avoid the DLC <-> JAX numpy version conflict).
        from pathlib import Path as _P
        pose_py = _P.cwd() / ".venv-pose" / "bin" / "python"
        main_py = _P.cwd() / ".venv" / "bin" / "python"
        if pose_py.exists():
            py = str(pose_py)
        elif main_py.exists():
            py = str(main_py)
        else:
            py = sys.executable
        cmd = [py, "scripts/extract_keypoints.py", str(out)]
        print("[fetch] chaining ->", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
