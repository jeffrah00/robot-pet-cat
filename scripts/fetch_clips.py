#!/usr/bin/env python3
"""Download the cat motion clips listed in data/motion_clips_raw/manifest.yaml.

For each clip with `source: pexels` it hits the Pexels API and picks the
highest-quality MP4 below ~5 MB compressed (full-quality is overkill for pose
extraction). For `source: youtube` or `source: url` it uses yt-dlp.

Optional time crop via ffmpeg if the manifest entry has a `crop` block.

Usage:
    # one-time: get a Pexels API key at https://www.pexels.com/api/
    export PEXELS_API_KEY=xxxxx

    python scripts/fetch_clips.py
    python scripts/fetch_clips.py --only walk_outdoor_gray,stretch_table
    python scripts/fetch_clips.py --skip-existing  # default; pass --force to redownload
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


def _load_manifest(manifest_path: Path) -> list[dict]:
    import yaml  # noqa: PLC0415
    with manifest_path.open() as f:
        data = yaml.safe_load(f)
    clips = data.get("clips", [])
    if not clips:
        raise SystemExit(f"No clips in {manifest_path}")
    return clips


def _pexels_video_url(video_id: str, api_key: str) -> str:
    """Hit the Pexels API for a video and return a sensible MP4 URL."""
    req = urllib.request.Request(
        f"{PEXELS_API_BASE}/{video_id}",
        headers={"Authorization": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Pexels API error for video {video_id}: HTTP {e.code} {e.reason}"
        ) from e

    files = payload.get("video_files", [])
    if not files:
        raise RuntimeError(f"Pexels video {video_id} has no video_files in response")

    # Prefer 720p (sweet spot for DLC; higher res is wasted compute).
    def score(f: dict) -> tuple[int, int]:
        h = f.get("height") or 0
        # rank distance from 720
        return (abs(h - 720), -h)

    files = sorted(files, key=score)
    return files[0]["link"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_clips]   downloading {url} -> {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "robot-pet-cat/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def _crop(src: Path, dest: Path, start_s: float, end_s: float) -> None:
    """Re-encode the clip cropped to [start_s, end_s] via ffmpeg."""
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


def fetch_one(clip: dict, raw_dir: Path, force: bool) -> tuple[bool, str]:
    """Download a single manifest entry. Returns (ok, message)."""
    cid = clip["id"]
    source = clip["source"]
    out_path = raw_dir / f"{cid}.mp4"

    if out_path.exists() and not force:
        return True, f"skip (exists)"

    if source == "pexels":
        api_key = os.environ.get("PEXELS_API_KEY")
        if not api_key:
            return False, "PEXELS_API_KEY env var not set (https://www.pexels.com/api/)"
        try:
            url = _pexels_video_url(str(clip["source_id_or_url"]), api_key)
        except RuntimeError as e:
            return False, str(e)
        try:
            tmp_path = raw_dir / f"{cid}.raw.mp4"
            _download(url, tmp_path)
        except Exception as e:  # noqa: BLE001
            return False, f"download failed: {e}"
    elif source in ("youtube", "url"):
        if shutil.which("yt-dlp") is None:
            return False, "yt-dlp not installed (pip install yt-dlp)"
        tmp_path = raw_dir / f"{cid}.raw.mp4"
        cmd = ["yt-dlp", "-f", "mp4", "-o", str(tmp_path), clip["source_id_or_url"]]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
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
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated list of clip IDs to fetch (default: all).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Redownload even if the .mp4 already exists.",
    )
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
        ok, msg = fetch_one(clip, args.out, args.force)
        print(f"[fetch_clips]   {'OK ' if ok else 'FAIL'}: {msg}")
        n_ok += int(ok)
        n_fail += int(not ok)

    print(f"[fetch_clips] done: {n_ok} ok, {n_fail} failed")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
