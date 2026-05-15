# Cat-video data collection

## Sources, in order of preference

1. **Pexels** — free for commercial use, no auth required for browsing.
   Search: https://www.pexels.com/search/videos/cat/
   API: https://www.pexels.com/api/documentation/
2. **Pixabay** — CC0. Search: https://pixabay.com/videos/search/cat/
3. **Wikimedia Commons** — public domain or CC. Smaller pool but high quality.
4. **Curated CC-BY YouTube channels** — *only* channels that explicitly mark
   videos CC-BY. Use `yt-dlp --match-filter "license=Creative Commons Attribution"`.
5. **Your own cat footage**, if applicable.

**Do not** scrape arbitrary YouTube, Instagram, or TikTok. Standard ToS prohibit
it and YouTube's "fair use for training" status is contested. Stick to licensed
sources.

## Filter pipeline

Each clip needs to pass:

1. **Single cat** in frame for ≥80% of duration (run a YOLOv8 cat detector).
2. **≥3 seconds, ≤30 seconds**. Longer clips get cut into 10-second chunks.
3. **Bounding box ≥ 15% of frame area** — too-far cats yield bad pose.
4. **Not heavily edited** — drop clips with hard cuts >1/sec (use scene-change
   detection).
5. **Reasonable lighting** — drop if mean luminance is too low or saturated.

## Pose extraction

Use MMPose's `animalpose-hrnet-w48` (or DeepLabCut if MMPose is finicky).
Outputs 20 keypoints per frame: head, ears, eyes, nose, neck, withers, hips,
four paws (heel + toe), tail base, tail tip.

Save per-clip as a `.npz` with shape `(T, 20, 3)` — last channel is confidence.

## Behavior labels

Two complementary approaches; do both, intersect for high confidence.

**Heuristic (cheap):**
- Stand vs. move: speed of withers keypoint < 0.05 m/s → standing.
- Crouch: hips lower than withers by >20%.
- Pounce: short burst of acceleration after a crouch.
- Groom: face keypoints close to paws for >2 sec.
- Sleep: minimal motion + body horizontal for >5 sec.

**Cluster (richer):**
- Embed each clip's pose sequence with a small temporal CNN.
- HDBSCAN over the embeddings.
- Manually label the resulting clusters.

## Licensing & datasheet

Every clip carries a sidecar JSON:

```json
{
  "id": "pexels_4827593",
  "source": "pexels",
  "url": "https://www.pexels.com/video/...",
  "license": "Pexels License",
  "attribution": "Video by Name",
  "duration_s": 12.4,
  "ingested_at": "2026-05-20T18:31:00Z"
}
```

The dataset on HF Hub gets a datasheet (`datasheet.md`) covering source mix,
license breakdown, biases (overrepresentation of indoor housecats vs. outdoor,
breed skew, etc.), and known failure modes.

## Storage layout

```
data/
├── raw/
│   ├── pexels/
│   │   ├── 4827593.mp4
│   │   └── 4827593.json
│   └── pixabay/
├── processed/
│   ├── pose/
│   │   └── 4827593.npz
│   └── labels/
│       └── 4827593.yaml
└── manifests/
    └── train.jsonl
```

Raw clips go to Cloudflare R2. Processed (pose + labels) goes to Hugging Face.
