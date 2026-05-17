#!/usr/bin/env python3
"""Run DeepLabCut SuperAnimal-Quadruped on a video and write our keypoint JSON.

Usage:
    python scripts/extract_keypoints.py path/to/cat.mp4 \
        --out data/motion_clips_raw/cat.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot_pet_cat.pose_extraction import ExtractConfig, extract  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path, help="Path to the cat .mp4 clip.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output keypoint JSON (default: data/motion_clips_raw/<stem>.json).")
    p.add_argument("--superanimal", default="superanimal_quadruped",
                   help="DLC SuperAnimal variant.")
    p.add_argument("--model-name", default="hrnet_w32", dest="model_name")
    p.add_argument("--detector-name", default="fasterrcnn_resnet50_fpn_v2", dest="detector_name")
    p.add_argument("--pcutoff", type=float, default=0.4,
                   help="Drop keypoints below this confidence.")
    p.add_argument("--body-length-m", type=float, default=0.45, dest="body_length_m",
                   help="Assumed cat withers->hips length in meters.")
    p.add_argument("--floor-y", type=float, default=None,
                   help="Image y-pixel of the floor; inferred from lowest paw if omitted.")
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                   help="DLC video inference batch size. 8 is a good GPU default; "
                        "lower if you hit OOM, higher if you have headroom.")
    args = p.parse_args()

    out = args.out or (Path("data/motion_clips_raw") / f"{args.video.stem}.json")
    cfg = ExtractConfig(
        video_path=args.video,
        out_json=out,
        superanimal=args.superanimal,
        model_name=args.model_name,
        detector_name=args.detector_name,
        pcutoff=args.pcutoff,
        cat_body_length_m=args.body_length_m,
        floor_y_px=args.floor_y,
        batch_size=args.batch_size,
    )
    written = extract(cfg)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
