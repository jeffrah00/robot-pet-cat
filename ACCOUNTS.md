# Accounts to create

Checklist of every external service this project depends on. Five required
accounts (well, three signups + one PAT + one optional API key), all free tier.

## Required

### 1. GitHub -- code hosting + Personal Access Token
- **You already have this.** Username: `jeffrah00`.
- **Empty repo:** create one named `robot-pet-cat` at https://github.com/new --
  no README, no .gitignore, no license. The scaffold has all of those.
- **Personal Access Token:** so the Cowork sandbox can push commits on your
  behalf via the REST API (no more manual `git push`).
  1. Generate at https://github.com/settings/tokens/new
  2. Scope: `repo` (full control of private repos) or `public_repo` (public only)
  3. Pick an expiry that suits you (90 days is reasonable)
  4. Add to `.env`: `GITHUB_TOKEN=ghp_...`
- The `scripts/sandbox_push.py` script reads this token and pushes specific
  files on demand. See `docs/retargeting.md` and the script's `--help`.

### 2. Hugging Face Hub -- models + motion-clip dataset
- **Signup:** https://huggingface.co/join (free)
- **What it gives us:** versioned storage for the AMP-trained motion
  checkpoint, every skill checkpoint, the brain checkpoint, and the
  retargeted motion-clip dataset.
- **Storage estimate:** ~5-15 GB total for all artifacts combined. Free tier
  is fine; HF Pro ($9/mo) only if we want private datasets.
- **After signup:**
  1. Create an access token at https://huggingface.co/settings/tokens (Write scope)
  2. Save it; you'll run `huggingface-cli login` and paste it.

### 3. Weights & Biases -- experiment tracking
- **Signup:** https://wandb.ai/site (free for personal use)
- **What it gives us:** training curves, hyperparameter sweeps, video logging
  of RL rollouts. Unlimited on the free personal tier.
- **After signup:** copy your API key from https://wandb.ai/authorize and run
  `wandb login`.

### 4. RunPod -- GPU compute
- **Signup:** https://www.runpod.io/console/signup
- **What it gives us:** A100 spot instances at ~$0.40-0.80/hr. Total project
  spend lands around $120-200.
- **Alternative:** [Lambda Labs](https://lambdalabs.com/) at ~$1.10/hr for
  A100 -- pricier but fewer preemptions for long runs.
- **After signup:** add $20 of credit. Generate an SSH key on your laptop
  (`ssh-keygen -t ed25519`) and paste the public key into
  Settings -> SSH Public Keys.

## Optional / later

### Pexels API -- for the automated clip-downloader
- **Signup:** https://www.pexels.com/api/new/ (free, 30 seconds)
- **What it gives us:** authenticated access to Pexels video metadata + direct
  .mp4 URLs. `scripts/fetch_clip.py` uses this to bypass Cloudflare's anti-bot
  challenge that yt-dlp's generic extractor hits.
- **Free tier:** 200 requests/hour, 20,000/month. Way more than we need.
- **After signup:** copy the key and add `PEXELS_API_KEY=...` to `.env`.

### NVIDIA Developer account -- only if we add a VLM-as-policy head later
- **Signup:** https://developer.nvidia.com/login (free)
- Needed for Isaac Lab. Out of scope for v1.

## After accounts exist -- push the repo (one time)

The `sandbox_push.py` script can update files on an existing repo, but it can't
do the very first push that creates `main`. Bootstrap once from your laptop:

```bash
cd "Robot pet cat"
git remote add origin git@github.com:jeffrah00/robot-pet-cat.git
git branch -M main
git push -u origin main
```

Or HTTPS:
```bash
git remote add origin https://github.com/jeffrah00/robot-pet-cat.git
```

After this initial push, all subsequent updates from the Cowork sandbox happen
via `scripts/sandbox_push.py`. You only ever pull from your end:
```powershell
cd "C:\Users\jeffr\Documents\Claude\Projects\Robot pet cat"
git pull
```

The `scripts/init_git.ps1` / `scripts/init_git.sh` helpers cover the bootstrap;
they print the remote-add command after the first commit.

## Local secrets

Copy `.env.example` to `.env` (already gitignored) and fill in the keys as
you create each account:

```bash
cp .env.example .env
# then edit .env
```
