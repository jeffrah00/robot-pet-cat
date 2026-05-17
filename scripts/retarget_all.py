#!/usr/bin/env python3
"""Run pose extraction + retargeting on every clip in the manifest.

Pipeline per clip:
  1. <id>.mp4 -> DLC SuperAnimal pose extraction (in the dlc conda env)
     -> <id>_keypoints.json
  2. <id>_keypoints.json -> our retargeter -> data/motion_clips/<id>.npz

Failure is non-fatal: a bad clip just doesn't produce an .npz. The summary
at the end tells you what worked.

Usage:
    python scripts/retarget_all.py
    python scripts/retarget_all.py --only walk_outdoor_gray,stretch_table
    python scripts/retarget_all.py --skip-pose-extraction
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _load_manifest(path: Path) -> list[dict]:
    import yaml
    with path.open() as f:
        return yaml.safe_load(f).get("clips", [])


def _run_pose_extraction(mp4_path: Path, keypoints_path: Path) -> tuple[bool, str]:
    """Invoke DLC via the dlc conda env (extract_keypoints.py runs SuperAnimal)."""
    if shutil.which("micromamba"):
        runner = ["micromamba", "run", "-n", "dlc"]
    elif shutil.which("conda"):
        runner = ["conda", "run", "-n", "dlc"]
    else:
        return False, "micromamba/conda not on PATH; can't access the dlc env"

    cmd = runner + [
        "python", "scripts/extract_keypoints.py",
        "--video", str(mp4_path),
        "--out", str(keypoints_path),
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        return False, f"DLC extract returned {e.returncode}"
    if not keypoints_path.is_file():
        return False, "DLC extract did not produce keypoints json"
    return True, "ok"


def _run_retarget(keypoints_path: Path, out_npz: Path) -> tuple[bool, str]:
    """Compose load_keypoints_json -> retarget_clip -> write_npz."""
    from robot_pet_cat.retargeting import (
        load_keypoints_json, retarget_clip, write_npz,
    )
    try:
        clip = load_keypoints_json(keypoints_path)
        data = retarget_clip(clip)
        write_npz(out_npz, data)
    except Exception as e:  # noqa: BLE001
        return False, f"retarget failed: {type(e).__name__}: {e}"
    if not out_npz.is_file():
        return False, "retarget did not produce npz"
    return True, "ok"


def process_clip(
    clip: dict,
    raw_dir: Path,
    npz_dir: Path,
    skip_pose: bool,
) -> tuple[bool, str]:
    cid = clip["id"]
    mp4_path = raw_dir / f"{cid}.mp4"
    keypoints_path = raw_dir / f"{cid}_keypoints.json"
    out_npz = npz_dir / f"{cid}.npz"

    if not mp4_path.is_file():
        return False, f"missing {mp4_path} (run fetch_clips.py first)"

    if not skip_pose or not keypoints_path.is_file():
        ok, msg = _run_pose_extraction(mp4_path, keypoints_path)
        if not ok:
            return False, f"pose extraction: {msg}"

    ok, msg = _run_retarget(keypoints_path, out_npz)
    if not ok:
        return False, f"retarget: {msg}"
    return True, f"ok ({out_npz.stat().st_size // 1024} KiB)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/motion_clips_raw/manifest.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("data/motion_clips_raw"))
    parser.add_argument("--out", type=Path, default=Path("data/motion_clips"))
    parser.add_argument("--only", default=None, help="Comma-separated clip IDs to process.")
    parser.add_argument("--skip-pose-extraction", action="store_true",
                        help="Reuse existing <id>_keypoints.json if present.")
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"[retarget_all] no manifest at {args.manifest}", file=sys.stderr)
        return 1

    clips = _load_manifest(args.manifest)
    if args.only:
        wanted = set(args.only.split(","))
        clips = [c for c in clips if c["id"] in wanted]
        if not clips:
            print(f"[retarget_all] --only matched zero clips ({sorted(wanted)})")
            return 1

    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for clip in clips:
        cid = clip["id"]
        print(f"[retarget_all] {cid} ({clip.get('behavior', '?')}):")
        ok, msg = process_clip(clip, args.raw, args.out, args.skip_pose_extraction)
        print(f"[retarget_all]   {'OK ' if ok else 'FAIL'}: {msg}")
        results.append((cid, ok, msg))

    print()
    print("[retarget_all] === summary ===")
    n_ok = sum(1 for _cid, ok, _msg in results if ok)
    n_fail = len(results) - n_ok
    for cid, ok, msg in results:
        flag = "OK  " if ok else "FAIL"
        print(f"  {flag}  {cid:30s}  {msg}")
    print(f"  total: {n_ok} ok, {n_fail} failed -> {args.out}/")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
