"""Extract 2D cat keypoints from a video using OpenPifPaf's animal-pose model.

This bridges raw .mp4 -> our 12-keypoint JSON schema consumed by the retargeting
module.

Why OpenPifPaf and not DLC SuperAnimal:
  - DLC 2.3.x pins tables==3.8.0, which has no Python 3.12 wheel and fails to
    compile blosc2 from source. We'd have to downgrade to 3.11.
  - OpenPifPaf installs cleanly on 3.11/3.12, has an animal-pose plugin
    explicitly trained on cats (alongside dogs/sheep/horses/cows), and is
    PyTorch-only with no HDF5 deps.

Trade-off: OpenPifPaf's animal pose model is somewhat less accurate than DLC's
SuperAnimal-Quadruped (smaller training set), but for AMP's distribution-
matching use case the gap doesn't matter much.

Alternatives if you need to swap the upstream model:
  - DeepLabCut SuperAnimal-Quadruped (heavy install, Python <=3.11)
  - MMPose AnimalPose
  - BARC for true 3D pose
The retargeting module doesn't care which detector you used, only the JSON
schema.
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

# OpenPifPaf's animal-pose model uses the AnimalPose dataset's 20-keypoint set:
#  0 L_Eye    1 R_Eye    2 L_EarBase  3 R_EarBase
#  4 Nose     5 Throat   6 TailBase   7 Withers
#  8 L_F_Elbow  9 R_F_Elbow  10 L_B_Elbow  11 R_B_Elbow
# 12 L_F_Knee  13 R_F_Knee  14 L_B_Knee   15 R_B_Knee
# 16 L_F_Paw   17 R_F_Paw   18 L_B_Paw    19 R_B_Paw
#
# Mapping our 12 -> indices in this 20-keypoint output.
# We deliberately reuse TailBase for both `hips` and `tail_base`: the
# retargeting math wants a stable rear-trunk reference, and TailBase is the
# closest anatomical match. Any subtle offset is absorbed by the AMP
# discriminator.
OPENPIFPAF_INDEX_MAP = {
    "nose":      4,
    "withers":   7,
    "hips":      6,    # TailBase used as hips proxy
    "tail_base": 6,
    "fl_paw":    16,
    "fr_paw":    17,
    "rl_paw":    18,
    "rr_paw":    19,
    "fl_elbow":  8,
    "fr_elbow":  9,
    "rl_knee":   14,
    "rr_knee":   15,
}


@dataclass
class ExtractConfig:
    video_path: Path
    out_json: Path
    checkpoint: str = "shufflenetv2k30-animalpose"
    pcutoff: float = 0.4              # drop keypoints with confidence below this
    sample_every_nth: int = 1         # 1 = every frame, 2 = every other, etc.
    cat_body_length_m: float = 0.45   # used to auto-compute scale_m_per_px
    floor_y_px: float | None = None   # if None, inferred from lowest paw y


def _load_predictor(checkpoint: str):
    """Lazy import + load. Importing openpifpaf is slow (several seconds)."""
    import openpifpaf  # noqa: PLC0415

    print(f"[pose] loading OpenPifPaf checkpoint: {checkpoint}")
    return openpifpaf.Predictor(checkpoint=checkpoint)


def get_video_meta(video_path: Path) -> tuple[float, int, int, int]:
    """Return (fps, image_height_px, image_width_px, n_frames) by peeking at the video."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 couldn't open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, h, w, n


def _pick_best_cat(predictions, frame_idx: int) -> object | None:
    """If multiple animals detected in one frame, keep the highest-score one.

    OpenPifPaf returns predictions sorted by score descending, so we can just
    take the first; but be defensive in case that contract changes.
    """
    if not predictions:
        return None
    return max(predictions, key=lambda a: getattr(a, "score", 0.0))


def run_openpifpaf(cfg: ExtractConfig) -> tuple[np.ndarray, float, int, int]:
    """Iterate over frames, run OpenPifPaf, return (T, 20, 3) keypoints.

    Returns:
        keypoints_20: (T, 20, 3) array of (x, y, confidence) in image pixels;
                      NaN where no animal was detected
        fps: video fps
        height_px: image height
        width_px: image width
    """
    import cv2  # noqa: PLC0415

    cfg.video_path = Path(cfg.video_path).resolve()
    if not cfg.video_path.exists():
        raise FileNotFoundError(cfg.video_path)

    fps, h, w, n_frames = get_video_meta(cfg.video_path)
    predictor = _load_predictor(cfg.checkpoint)

    cap = cv2.VideoCapture(str(cfg.video_path))
    out: list[np.ndarray] = []
    frame_idx = 0
    sampled = 0

    print(f"[pose] processing {n_frames} frames "
          f"(sample_every_nth={cfg.sample_every_nth})")

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % cfg.sample_every_nth == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                predictions, _gt, _meta = predictor.numpy_image(rgb)
                ann = _pick_best_cat(predictions, frame_idx)
                if ann is None:
                    out.append(np.full((20, 3), np.nan, dtype=np.float64))
                else:
                    data = np.asarray(ann.data, dtype=np.float64)
                    if data.shape != (20, 3):
                        # Defensive: pad/crop to 20.
                        kps = np.full((20, 3), np.nan)
                        n_kp = min(data.shape[0], 20)
                        kps[:n_kp] = data[:n_kp]
                        data = kps
                    out.append(data)
                sampled += 1
                if sampled % 30 == 0:
                    print(f"[pose] {sampled} frames processed")
            frame_idx += 1
    finally:
        cap.release()

    print(f"[pose] done: {sampled} frames")
    arr = np.stack(out, axis=0) if out else np.zeros((0, 20, 3))
    # Effective fps is reduced by the sampling stride.
    eff_fps = fps / cfg.sample_every_nth
    return arr, eff_fps, h, w


def openpifpaf_to_project_keypoints(
    kps_20: np.ndarray,
    pcutoff: float,
) -> np.ndarray:
    """Convert OpenPifPaf's (T, 20, 3) output to our (T, 12, 2) array.

    Confidence < pcutoff -> NaN.
    """
    T = kps_20.shape[0]
    out = np.full((T, len(PROJECT_KP_NAMES), 2), np.nan, dtype=np.float64)
    for our_idx, name in enumerate(PROJECT_KP_NAMES):
        src_idx = OPENPIFPAF_INDEX_MAP[name]
        xs = kps_20[:, src_idx, 0]
        ys = kps_20[:, src_idx, 1]
        conf = kps_20[:, src_idx, 2]
        mask = (conf >= pcutoff) & np.isfinite(xs) & np.isfinite(ys)
        out[mask, our_idx, 0] = xs[mask]
        out[mask, our_idx, 1] = ys[mask]
    return out


def fill_missing_linear(kps: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN frames per keypoint per coord.

    Edge NaNs are filled by nearest-neighbour. If a keypoint is NaN in every
    frame, it's left as NaN.
    """
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
    """Floor y in image space = max paw y observed (largest image-y == lowest in image)."""
    paw_y = kps[:, 4:8, 1]
    return float(np.nanmax(paw_y))


def infer_scale_m_per_px(kps: np.ndarray, body_length_m: float) -> float:
    """Estimate world scale from the median withers->hips pixel distance."""
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
    """End-to-end: video -> OpenPifPaf -> our JSON keypoints schema."""
    raw, fps, h, _w = run_openpifpaf(cfg)
    if raw.shape[0] == 0:
        raise RuntimeError(f"No frames decoded from {cfg.video_path}")

    kps_partial = openpifpaf_to_project_keypoints(raw, cfg.pcutoff)
    nan_frac_per_kp = np.isnan(kps_partial[..., 0]).mean(axis=0)
    print("[pose] NaN fraction per keypoint (before fill):")
    for name, frac in zip(PROJECT_KP_NAMES, nan_frac_per_kp, strict=True):
        flag = "  WARN" if frac > 0.5 else ""
        print(f"        {name:>10}: {frac:.2%}{flag}")

    kps = fill_missing_linear(kps_partial)
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

    print(f"[pose] effective fps={fps:.2f}  image_height={h}px")
    print(f"[pose] inferred scale = {scale:.4f} m/px "
          f"(assuming cat body length {cfg.cat_body_length_m} m)")
    print(f"[pose] inferred floor_y = {floor_y:.1f} px")

    return write_project_json(
        out_path=cfg.out_json,
        kps=kps,
        fps=fps,
        image_height_px=h,
        scale_m_per_px=scale,
        floor_y_px=floor_y,
        source=f"openpifpaf/{cfg.checkpoint}",
    )
