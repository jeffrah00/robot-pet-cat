#!/usr/bin/env python3
"""Download one Pexels video into data/motion_clips_raw/ using yt-dlp.

yt-dlp's Pexels extractor handles the URL -> direct-mp4 resolution for us, so
no API key is needed for one-off pulls. For high-volume scraping later we'll
add a Pexels-API path with a key (200 req/hr free tier).

Usage:
    python scripts/fetch_clip.py https://www.pexels.com/video/gray-cat-walking-outdoors-on-a-cloudy-day-34482817/
    python scripts/fetch_clip.py 34482817                               # numeric id also works
    python scripts/fetch_clip.py <url> --out data/motion_clips_raw/cat1.mp4
    python scripts/fetch_clip.py <url> --then-extract                   # chain into pose extraction
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_url(arg: str) -> str:
    """Accept either a full Pexels URL or just the numeric video ID."""
    if arg.isdigit():
        return f"https://www.pexels.com/video/{arg}/"
    return arg


def slug_from_url(url: str) -> str:
    """Strip Pexels URL to a clean filename slug."""
    m = re.search(r"/video/([^/]+?)-(\d+)/?", url)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"/video/(\d+)/?", url)
    if m:
        return f"pexels-{m.group(1)}"
    return "clip"


def have_yt_dlp() -> bool:
    return shutil.which("yt-dlp") is not None


def download(url: str, out_path: Path) -> Path:
    """Drive yt-dlp to fetch the highest mp4 the page exposes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--no-playlist",
        "--format", "bestvideo[ext=mp4]/best[ext=mp4]/best",
        "--output", str(out_path),
        url,
    ]
    print("[fetch]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not out_path.exists():
        raise RuntimeError(f"yt-dlp claimed success but {out_path} is missing")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="Pexels URL or numeric video id.")
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output mp4 path (default: data/motion_clips_raw/<slug>.mp4).",
    )
    p.add_argument(
        "--then-extract", action="store_true",
        help="Chain into scripts/extract_keypoints.py after the download.",
    )
    args = p.parse_args()

    if not have_yt_dlp():
        print("yt-dlp not found. Install with: pip install -e \".[retargeting]\"")
        return 2

    url = resolve_url(args.source)
    out = args.out or Path("data/motion_clips_raw") / f"{slug_from_url(url)}.mp4"

    download(url, out)
    print(f"[fetch] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    if args.then_extract:
        extract_cmd = [
            sys.executable, "scripts/extract_keypoints.py", str(out),
        ]
        print("[fetch] chaining ->", " ".join(extract_cmd))
        subprocess.run(extract_cmd, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
