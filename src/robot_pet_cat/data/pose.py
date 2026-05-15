"""Extract 2D cat keypoints from clips using MMPose AnimalPose (or DeepLabCut)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PoseConfig:
    model: str = "mmpose"  # mmpose | dlc
    weights: str = "animalpose-hrnet-w48"  # default for cats/dogs


def extract(clip_path: Path, cfg: PoseConfig) -> Path:
    """Run pose extraction on a clip; write a .npz of keypoints next to it."""
    raise NotImplementedError("Phase 3: MMPose AnimalPose inference loop.")
