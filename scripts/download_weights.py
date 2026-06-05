"""Download released robot-pet-cat weights from GitHub.

Usage:
    python scripts/download_weights.py

Downloads all released model weights into models/ and checkpoints/brain/.
"""

from __future__ import annotations
import urllib.request
from pathlib import Path

REPO = "jeffrah00/robot-pet-cat"
RAW  = f"https://github.com/{REPO}/raw/main"

WEIGHTS = {
    "models/go1_walker_v0.pt":          f"{RAW}/models/go1_walker_v0.pt",
    "models/mjlab_go1_walker_normal.pt": f"{RAW}/models/mjlab_go1_walker_normal.pt",
    "models/mjlab_go1_crouch.pt":        f"{RAW}/models/mjlab_go1_crouch.pt",
    "models/mjlab_go1_lie_belly.pt":     f"{RAW}/models/mjlab_go1_lie_belly.pt",
    "checkpoints/brain/brain_4skill_v1.zip": f"{RAW}/checkpoints/brain/brain_4skill_v1.zip",
}

def download(dest: str, url: str) -> None:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"  already exists: {dest}")
        return
    print(f"  downloading {dest} ...")
    urllib.request.urlretrieve(url, path)
    print(f"  saved {path.stat().st_size // 1024} KB")

if __name__ == "__main__":
    for dest, url in WEIGHTS.items():
        download(dest, url)
    print("Done.")
