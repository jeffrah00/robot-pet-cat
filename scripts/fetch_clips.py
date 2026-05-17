#!/usr/bin/env python3
"""Download the cat motion clips listed in data/motion_clips_raw/manifest.yaml.

Two download paths for Pexels:
  1. Pexels API (requires PEXELS_API_KEY): cleaner, picks 720p MP4 directly.
  2. yt-dlp fallback: no API key needed, uses Pexels URL pattern.
     Triggered when --no-api is passed OR when the API returns 401/403/404.

For source: youtube or source: url, yt-dlp is always used.

Optional time crop via ffmpeg if the manifest entry has a `crop` block.

Cloudflare note: Pexels is behind Cloudflare and blocks bot-shaped
User-Agents with 403, even with a valid API key. We send a Chrome-shaped
UA + standard Accept headers so the request looks like a normal web client.

Usage:
    export PEXELS_API_KEY=xxxxx          # get one at https://www.pexels.com/api/
    python scripts/fetch_clips.py

    pip install yt-dlp                   # alternative path, no key needed
    python scripts/fetch_clips.py --no-api

    python scripts/fetch_clips.py --only walk_outdoor_gray,stretch_table
    python scripts/fetch_clips.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PEXELS_API_BASE = "https://api.pexels.com/videos/videos"
PEXELS_PAGE_URL = "https://www.pexels.com/video/{}/"

# Cloudflare-friendly headers. Pexels' bot detection 403s anything that
# looks like a custom UA (we previously sent "robot-pet-cat/1.0" and got
# 403 on every request; curl/7.* with a real API key worked fine).
CLOUDFLARE_OK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _load_manifest(manifest_path: Path) -> list[dict]:
    import yaml
    with manifest_path.open() as f:
        data = yaml.safe_load(f)
    clips = data.get("clips", [])
    if not clips:
        raise SystemExit(f"No clips in {manifest_path}")
    return clips


def _pexels_video_url_via_api(video_id: str, api_key: str) -> str:
    """Hit the Pexels API for a video and return a sensible MP4 URL."""
    headers = dict(CLOUDFLARE_OK_HEADERS)
    headers["Authorization"] = api_key
    headers["Accept"] = "application/json, */*;q=0.5"
    req = urllib.request.Request(f"{PEXELS_API_BASE}/{video_id}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Pexels API HTTP {e.code} {e.reason} for video {video_id}"
        ) from e

    files = payload.get("video_files", [])
    if not files:
        raise RuntimeError(f"Pexels video {video_id} has no video_files in response")

    def score(f: dict) -> tuple[int, int]:
        h = f.get("height") or 0
        return (abs(h - 720), -h)

    files = sorted(files, key=score)
    return files[0]["link"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_clips]   downloading {url[:80]}{'...' if len(url) > 80 else ''} -> {dest.name}")
    req = urllib.request.Request(url, headers=CLOUDFLARE_OK_HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def _yt_dlp_download(url: str, dest: Path) -> None:
    """Use yt-dlp to download from any supported site (incl. Pexels)."""
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not installed (pip install yt-dlp)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_clips]   yt-dlp {url}")
    cmd = [
        "yt-dlp",
        "-q", "--no-warnings",
        "-f", "mp4/bv*+ba/b",
        "--merge-output-format", "mp4",
        "-o", str(dest),
        url,
    ]
    subprocess.check_call(cmd)


def _crop(src: Path, dest: Path, start_s: float, end_s: float) -> None:
    if shutil.which("ffmpeg") is None:
        print(f"[fetch_clips]   WARNING: ffmpeg not installed; skipping crop on {src.name}")
        if src != dest:
            shutil.copy(src, dest)
        return
    print(f"[fetch_clips]   cropping {src.name} -> {dest.name} [{start_s}..{end_s}]")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", str(src),
        "-ss", str(start_s),
        "-to", str(end_s),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-an",
        str(dest),
    ]
    subprocess.check_call(cmd)


def fetch_one(clip: dict, raw_dir: Path, force: bool, prefer_api: bool) -> tuple[bool, str]:
    """Download a single manifest entry. Returns (ok, message)."""
    cid = clip["id"]
    source = clip["source"]
    out_path = raw_dir / f"{cid}.mp4"

    if out_path.exists() and not force:
        return True, "skip (exists)"

    tmp_path = raw_dir / f"{cid}.raw.mp4"

    if source == "pexels":
        pid = str(clip["source_id_or_url"])
        downloaded_via = None

        if prefer_api and os.environ.get("PEXELS_API_KEY"):
            try:
                url = _pexels_video_url_via_api(pid, os.environ["PEXELS_API_KEY"])
                _download(url, tmp_path)
                downloaded_via = "api"
            except (RuntimeError, urllib.error.URLError) as e:
                print(f"[fetch_clips]   API failed ({e}); falling back to yt-dlp")

        if downloaded_via is None:
            page_url = PEXELS_PAGE_URL.format(pid)
            try:
                _yt_dlp_download(page_url, tmp_path)
                downloaded_via = "yt-dlp"
            except (RuntimeError, subprocess.CalledProcessError) as e:
                return False, f"both API and yt-dlp failed: {e}"

    elif source in ("youtube", "url"):
        try:
            _yt_dlp_download(clip["source_id_or_url"], tmp_path)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            return False, f"yt-dlp failed: {e}"
    else:
        return False, f"unknown source {source!r}"

    crop = clip.get("crop") or {}
    if crop:
        try:
            _crop(tmp_path, out_path, float(crop["start_s"]), float(crop["end_s"]))
            tmp_path.unlink(missing_ok=True)
        except subprocess.CalledProcessError as e:
            return False, f"crop failed: {e}"
    else:
        tmp_path.rename(out_path)

    return True, f"ok ({out_path.stat().st_size // 1024} KiB)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/motion_clips_raw/manifest.yaml"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/motion_clips_raw"))
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of clip IDs to fetch.")
    parser.add_argument("--force", action="store_true",
                        help="Redownload even if the .mp4 already exists.")
    parser.add_argument("--no-api", action="store_true",
                        help="Skip Pexels API and use yt-dlp directly.")
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"[fetch_clips] no manifest at {args.manifest}", file=sys.stderr)
        return 1

    clips = _load_manifest(args.manifest)
    if args.only:
        wanted = set(args.only.split(","))
        clips = [c for c in clips if c["id"] in wanted]
        if not clips:
            print(f"[fetch_clips] --only matched zero clips ({sorted(wanted)})")
            return 1

    args.out.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    for clip in clips:
        cid = clip["id"]
        print(f"[fetch_clips] {cid} ({clip.get('behavior','?')}):")
        ok, msg = fetch_one(clip, args.out, args.force, prefer_api=not args.no_api)
        print(f"[fetch_clips]   {'OK ' if ok else 'FAIL'}: {msg}")
        n_ok += int(ok)
        n_fail += int(not ok)

    print(f"[fetch_clips] done: {n_ok} ok, {n_fail} failed")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
