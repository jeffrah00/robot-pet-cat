#!/usr/bin/env python3
"""Run DeepLabCut SuperAnimal-Quadruped on a video and write our keypoint JSON.

Usage:
    python scripts/extract_keypoints.py path/to/cat.mp4 \
        --out data/motion_clips_raw/cat.json

The output JSON is consumed by `rpc retarget`. See docs/retargeting.md for the
full Phase 2a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project sources are under src/.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot_pet_cat.pose_extraction import ExtractConfig, extract  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path, help="Path to the cat .mp4 clip.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output keypoint JSON path (default: data/motion_clips_raw/<stem>.json).",
    )
    p.add_argument(
        "--pcutoff",
        type=float,
        default=0.4,
        help="Drop keypoints with confidence below this (0..1).",
    )
    p.add_argument(
        "--body-length-m",
        type=float,
        default=0.45,
        dest="body_length_m",
        help="Assumed cat withers-to-hips length in meters; used to auto-compute scale.",
    )
    p.add_argument(
        "--floor-y",
        type=float,
        default=None,
        help="Image y in pixels of the floor; if omitted, inferred from lowest paw.",
    )
    p.add_argument(
        "--superanimal",
        default="superanimal_quadruped",
        help="DLC SuperAnimal variant.",
    )
    p.add_argument("--model-name", default="hrnet_w32")
    p.add_argument("--detector-name", default="fasterrcnn_resnet50_fpn_v2")
    args = p.parse_args()

    out = args.out or (
        Path("data/motion_clips_raw") / f"{args.video.stem}.json"
    )
    cfg = ExtractConfig(
        video_path=args.video,
        out_json=out,
        superanimal=args.superanimal,
        model_name=args.model_name,
        detector_name=args.detector_name,
        pcutoff=args.pcutoff,
        cat_body_length_m=args.body_length_m,
        floor_y_px=args.floor_y,
    )
    written = extract(cfg)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
