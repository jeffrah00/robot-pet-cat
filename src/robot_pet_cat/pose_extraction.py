"""Extract 2D cat keypoints from a video using DeepLabCut SuperAnimal-Quadruped.

We tried OpenPifPaf first because it's lighter, but openpifpaf 0.13.x has
unfixable Python-3.11+ build issues (pins torch==1.12.1 at build time, no
wheels for modern Python). DLC SuperAnimal-Quadruped is the second choice:
heavier install, but the install path on Python 3.11 with setuptools<70 is
known-good. Trained on 40K+ animal images including cats.

Pipeline:
  - deeplabcut.video_inference_superanimal(...) on the clip
  - Read the resulting .h5 with pandas
  - Map SuperAnimal's 39-keypoint bodypart output to our 12-keypoint schema
  - Linearly interpolate over low-confidence frames
  - Infer pixel scale + floor y-coordinate
  - Write our project JSON for retargeting

The retargeting module doesn't care which detector you used; only the JSON
schema is important.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Our project's 12-keypoint convention (matches src/robot_pet_cat/retargeting.py).
PROJECT_KP_NAMES = [
    "nose", "withers", "hips", "tail_base",
    "fl_paw", "fr_paw", "rl_paw", "rr_paw",
    "fl_elbow", "fr_elbow", "rl_knee", "rr_knee",
]

# Map our 12 -> DLC SuperAnimal-Quadruped bodypart names.
# Lists of candidates are tried in order; the first one present in the DLC
# output wins. This tolerates small naming drift between DLC versions.
# SuperAnimal-Quadruped's 39 bodyparts include:
#   nose, upper_jaw, lower_jaw, mouth_end_right, mouth_end_left,
#   right_eye, right_earbase, right_earend, left_eye, left_earbase, left_earend,
#   neck_base, neck_end, throat_base, throat_end,
#   back_base, back_end, back_middle,
#   tail_base, tail_end,
#   front_left_thai, front_left_knee, front_left_paw,
#   front_right_thai, front_right_knee, front_right_paw,
#   back_left_thai, back_left_knee, back_left_paw,
#   back_right_thai, back_right_knee, back_right_paw,
#   belly_bottom, body_middle_right, body_middle_left
# (DLC calls the front-leg "elbow" a "knee" -- we map accordingly.)
SUPERANIMAL_KP_MAP: dict[str, list[str]] = {
    "nose":      ["nose"],
    "withers":   ["neck_end", "neck_base", "back_base"],
    "hips":      ["back_end", "back_middle"],
    "tail_base": ["tail_base"],
    "fl_paw":    ["front_left_paw"],
    "fr_paw":    ["front_right_paw"],
    "rl_paw":    ["back_left_paw"],
    "rr_paw":    ["back_right_paw"],
    "fl_elbow":  ["front_left_knee"],
    "fr_elbow":  ["front_right_knee"],
    "rl_knee":   ["back_left_knee"],
    "rr_knee":   ["back_right_knee"],
}


@dataclass
class ExtractConfig:
    video_path: Path
    out_json: Path
    superanimal: str = "superanimal_quadruped"
    model_name: str = "hrnet_w32"
    detector_name: str = "fasterrcnn_resnet50_fpn_v2"
    pcutoff: float = 0.4
    cat_body_length_m: float = 0.45
    floor_y_px: float | None = None


def get_video_meta(video_path: Path) -> tuple[float, int]:
    """Return (fps, image_height_px) by peeking at the video with OpenCV."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 couldn't open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, height


def run_superanimal(cfg: ExtractConfig):
    """Invoke DLC's video_inference_superanimal. Imports deeplabcut lazily so
    this module is importable on machines without DLC installed.

    Returns the DLC output dataframe (multi-indexed by bodypart x coord) plus
    the path to the produced .h5.
    """
    import deeplabcut  # noqa: PLC0415

    cfg.video_path = Path(cfg.video_path).resolve()
    if not cfg.video_path.exists():
        raise FileNotFoundError(cfg.video_path)

    print(f"[pose] running DLC SuperAnimal on {cfg.video_path}")
    deeplabcut.video_inference_superanimal(
        videos=[str(cfg.video_path)],
        superanimal_name=cfg.superanimal,
        model_name=cfg.model_name,
        detector_name=cfg.detector_name,
        video_adapt=False,
        pcutoff=cfg.pcutoff,
    )

    candidates = list(cfg.video_path.parent.glob(f"{cfg.video_path.stem}*.h5"))
    if not candidates:
        raise RuntimeError(f"DLC produced no .h5 next to {cfg.video_path}")
    h5_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[pose] loading DLC output {h5_path}")

    import pandas as pd  # noqa: PLC0415

    df = pd.read_hdf(h5_path)
    return df, h5_path


def _resolve_dlc_name(bodyparts: list[str], candidates: list[str]) -> str | None:
    """First candidate present. None if none match."""
    available = set(bodyparts)
    for c in candidates:
        if c in available:
            return c
    return None


def dlc_to_project_keypoints(df, pcutoff: float) -> tuple[np.ndarray, list[str]]:
    """Convert a DLC SuperAnimal dataframe to our (T, 12, 2) keypoint array.

    Low-confidence detections are replaced by NaN.
    """
    df = df.copy()
    df.columns = df.columns.droplevel(0)
    bodyparts = sorted({c[0] for c in df.columns})

    resolved: list[tuple[str, str]] = []
    for our_name in PROJECT_KP_NAMES:
        dlc_name = _resolve_dlc_name(bodyparts, SUPERANIMAL_KP_MAP[our_name])
        if dlc_name is None:
            raise KeyError(
                f"None of {SUPERANIMAL_KP_MAP[our_name]!r} present in DLC output. "
                f"Available: {bodyparts}"
            )
        resolved.append((our_name, dlc_name))

    n_frames = len(df)
    out = np.full((n_frames, len(PROJECT_KP_NAMES), 2), np.nan, dtype=np.float64)
    for i, (_our, dlc_name) in enumerate(resolved):
        x = df[(dlc_name, "x")].to_numpy(dtype=np.float64)
        y = df[(dlc_name, "y")].to_numpy(dtype=np.float64)
        likelihood = df[(dlc_name, "likelihood")].to_numpy(dtype=np.float64)
        mask = likelihood >= pcutoff
        out[mask, i, 0] = x[mask]
        out[mask, i, 1] = y[mask]
    return out, [r[1] for r in resolved]


def fill_missing_linear(kps: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN frames per keypoint per coord."""
    out = kps.copy()
    T, N, _ = out.shape
    for i in range(N):
        for c in range(2):
            v = out[:, i, c]
            good = ~np.isnan(v)
            if good.sum() == 0:
                continue
            idx = np.arange(T)
            v[~good] = np.interp(idx[~good], idx[good], v[good])
            out[:, i, c] = v
    return out


def infer_floor_y_px(kps: np.ndarray) -> float:
    """Floor y in image space = max paw y observed."""
    paw_y = kps[:, 4:8, 1]
    return float(np.nanmax(paw_y))


def infer_scale_m_per_px(kps: np.ndarray, body_length_m: float) -> float:
    """Estimate world scale from median withers->hips pixel distance."""
    withers = kps[:, 1, :]
    hips = kps[:, 2, :]
    d_px = np.linalg.norm(withers - hips, axis=1)
    d_px = d_px[~np.isnan(d_px)]
    if d_px.size == 0:
        raise RuntimeError("All withers/hips keypoints are NaN; can't infer scale.")
    median_px = float(np.median(d_px))
    if median_px <= 0:
        raise RuntimeError(f"Median withers->hips distance is {median_px}; bad data.")
    return body_length_m / median_px


def write_project_json(
    out_path: Path,
    kps: np.ndarray,
    fps: float,
    image_height_px: int,
    scale_m_per_px: float,
    floor_y_px: float,
    source: str,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        {"t": float(i / fps), "keypoints": kps[i].tolist()}
        for i in range(kps.shape[0])
    ]
    payload = {
        "fps": float(fps),
        "scale_m_per_px": float(scale_m_per_px),
        "floor_y_px": float(floor_y_px),
        "image_height_px": int(image_height_px),
        "keypoint_names": PROJECT_KP_NAMES,
        "source": source,
        "frames": frames,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def extract(cfg: ExtractConfig) -> Path:
    """End-to-end: video -> DLC SuperAnimal -> our JSON keypoints schema."""
    fps, height = get_video_meta(cfg.video_path)
    df, _ = run_superanimal(cfg)
    raw_kps, resolved = dlc_to_project_keypoints(df, cfg.pcutoff)

    print("[pose] resolved keypoint mapping:")
    for our_name, dlc_name in zip(PROJECT_KP_NAMES, resolved, strict=True):
        print(f"        {our_name:>10} <- {dlc_name}")

    kps = fill_missing_linear(raw_kps)
    if np.isnan(kps).any():
        raise RuntimeError(
            "Some keypoints are NaN even after interpolation. "
            "Try lowering --pcutoff or use a clearer clip."
        )

    scale = infer_scale_m_per_px(kps, cfg.cat_body_length_m)
    floor_y = (
        cfg.floor_y_px
        if cfg.floor_y_px is not None
        else infer_floor_y_px(kps)
    )

    print(f"[pose] fps={fps:.2f}  image_height={height}px")
    print(f"[pose] inferred scale = {scale:.4f} m/px "
          f"(assuming cat body length {cfg.cat_body_length_m} m)")
    print(f"[pose] inferred floor_y = {floor_y:.1f} px")

    return write_project_json(
        out_path=cfg.out_json,
        kps=kps,
        fps=fps,
        image_height_px=height,
        scale_m_per_px=scale,
        floor_y_px=floor_y,
        source=f"deeplabcut/{cfg.superanimal}/{cfg.model_name}",
    )
