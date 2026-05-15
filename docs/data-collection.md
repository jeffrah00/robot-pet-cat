# Motion-clip collection

The previous version of this doc described scraping tens of hours of cat
video. After the AMP refactor that pipeline is gone — we now need ~10-30
short clips total, for the AMP style discriminator. This is a one-day job.

## What we need

- **~10-30 clips, 3-10 seconds each.** Quality bar: visible cat, mostly side
  view, single cat, modest camera motion. Pose-extraction quality matters more
  than visual variety.
- **Behavior mix:** walking, sitting down, stretching, crouching, leaping,
  ambient idle motion. Aim for 2-4 clips per behavior.
- **Total dataset size:** maybe 100-300 MB. Fits trivially on the HF Hub.

We are training a style *discriminator*, not a classifier. The discriminator
only sees pose transitions, so visual aesthetics don't matter — only that
the underlying motion distribution is catty.

## Sources, in order of preference

1. **Pexels** — free for commercial use.
   https://www.pexels.com/search/videos/cat/
2. **Pixabay** — CC0. https://pixabay.com/videos/search/cat/
3. **Wikimedia Commons** — public-domain or CC.
4. **Curated CC-BY YouTube channels** — `yt-dlp --match-filter
   "license=Creative Commons Attribution"`.
5. **Your own cat footage**, if applicable.

Skip the rest. Standard YouTube/TikTok/Instagram ToS prohibit scraping;
EgoPet-style scraping is for academic-publication contexts, not personal
projects.

## Pose extraction → retargeting → AMP clips

The motion clip pipeline lives in `scripts/extract_clip_poses.py` (filled in
during Phase 2). Steps:

1. **3D pose** via [BARC](https://barc.is.tue.mpg.de/) (a SMAL variant tuned
   for dogs/cats). Outputs per-frame body+limb joint angles in a parametric
   skeleton.
2. **Retarget to Go2** via IK using [Peng et al.'s motion_imitation
   code](https://github.com/erwincoumans/motion_imitation) — adjust the
   skeleton-mapping config for the cat skeleton vs the original dog skeleton.
3. **Save as `.npz`** in `data/motion_clips/<clip_id>.npz` with shape
   `(T, n_dof, 2)` — joint positions and velocities for the AMP
   discriminator.

The retargeted clips will not look like the original cat motion played back
perfectly — Go2 has fewer DOF. They'll look like a Go2 doing a rough
approximation. That's fine; AMP wants distribution similarity, not
photographic fidelity.

## Licensing & datasheet

Every clip gets a sidecar JSON:

```json
{
  "id": "pexels_4827593",
  "source": "pexels",
  "url": "https://www.pexels.com/video/...",
  "license": "Pexels License",
  "attribution": "Video by Name",
  "duration_s": 6.4
}
```

The clip dataset on HF Hub (`jeffrah00/cat-motion-clips`) gets a short
datasheet covering sources and licenses.

## Storage layout

```
data/
├── motion_clips/                  # ~30 retargeted .npz reference trajectories
│   ├── walk_01.npz
│   ├── sit_01.npz
│   └── ...
└── motion_clips_raw/              # gitignored: original .mp4 + .json sidecars
    ├── pexels_4827593.mp4
    ├── pexels_4827593.json
    └── ...
```

The retargeted `.npz` files are what AMP trains on. The raw `.mp4` files are
kept locally for reproducibility but not pushed to HF.
