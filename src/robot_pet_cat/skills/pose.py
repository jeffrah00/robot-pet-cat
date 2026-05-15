"""DEPRECATED.

This file used to hold the cat-video pose-extraction pipeline. After the
project pivot to AMP + skills + brain (no large-scale video dataset needed),
pose extraction collapsed to a one-shot script for the ~10-30 motion clips
that seed AMP. That lives in `scripts/extract_clip_poses.py` now.

Keeping this file empty so old imports fail loudly rather than silently
loading stale code.
"""

raise ImportError(
    "robot_pet_cat.skills.pose was removed in the AMP refactor. "
    "Motion-clip pose extraction now lives in scripts/extract_clip_poses.py."
)
