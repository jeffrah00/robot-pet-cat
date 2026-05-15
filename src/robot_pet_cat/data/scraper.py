"""Cat-video scraper.

Only ingest sources we have a license for:
  - Pexels (free for commercial use, attribution appreciated): https://www.pexels.com/search/videos/cat/
  - Pixabay (CC0)
  - Public-domain / CC-licensed YouTube channels listed in `manifest.yaml`
  - The user's own footage

The scraper writes raw clips to `data/raw/<source>/<id>.mp4` plus a parallel
`data/raw/<source>/<id>.json` with metadata + license info. We keep raw clips on
Cloudflare R2; only the processed (pose-annotated, downsampled) clips go to
Hugging Face.

Filled in during Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScrapeJob:
    source: str  # pexels | pixabay | manifest
    out_dir: Path
    max_clips: int = 1000
    min_duration_s: float = 3.0
    max_duration_s: float = 30.0


def run(job: ScrapeJob) -> None:
    raise NotImplementedError("Phase 3: implement Pexels/Pixabay API + yt-dlp manifest.")
