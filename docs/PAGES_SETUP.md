# Enabling the project page

The site at `docs/index.html` is a single static HTML file. To publish it at
`https://jeffrah00.github.io/robot-pet-cat/`:

1. Go to **Settings → Pages** on the `jeffrah00/robot-pet-cat` repo.
2. Source: **Deploy from a branch**.
3. Branch: **main** · folder: **/docs**.
4. Save. The page will be live within a minute or so.

`docs/.nojekyll` is included so GitHub Pages serves files as-is rather than
running them through Jekyll.

## Video hosting note

The page references video files via
`https://github.com/jeffrah00/robot-pet-cat/raw/main/renders/*.mp4`. This
works for files under ~25 MB. The 10-min hero rollout
(`renders/brain_4skill_10min.mp4`) is ~23.7 MB and should serve fine, but
if larger versions are added later, switch to Git LFS or upload to a
GitHub Release and update the `<source src=...>` URLs.

## Local preview

```bash
python3 -m http.server 8000 --directory docs
# open http://localhost:8000
```

The hero video and the skill grid will only load if you're online (they pull
from `github.com/.../raw/main/`).
