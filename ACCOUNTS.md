# Accounts to create

Checklist of every external service this project depends on. Four required
accounts plus one optional, all free tier.

## Required

### 1. GitHub — code hosting
- **You already have this.** Username: `jeffrah00`.
- **Next step:** create an empty repo named `robot-pet-cat` at
  https://github.com/new -- no README, no .gitignore, no license. The scaffold
  has all of those.
- Push instructions are at the bottom of this file (or run
  `scripts/init_git.ps1` from PowerShell).

### 2. Hugging Face Hub — models + motion-clip dataset
- **Signup:** https://huggingface.co/join (free)
- **What it gives us:** versioned storage for the AMP-trained motion
  checkpoint, every skill checkpoint, the brain checkpoint, and the
  retargeted motion-clip dataset.
- **Storage estimate:** ~5-15 GB total for all artifacts combined. Free tier
  is fine; HF Pro ($9/mo) only if we want private datasets.
- **After signup:**
  1. Create an access token at https://huggingface.co/settings/tokens (Write scope)
  2. Save it; you'll run `huggingface-cli login` and paste it.

### 3. Weights & Biases — experiment tracking
- **Signup:** https://wandb.ai/site (free for personal use)
- **What it gives us:** training curves, hyperparameter sweeps, video logging
  of RL rollouts. Unlimited on the free personal tier.
- **After signup:** copy your API key from https://wandb.ai/authorize and run
  `wandb login`.

### 4. RunPod — GPU compute
- **Signup:** https://www.runpod.io/console/signup
- **What it gives us:** A100 spot instances at ~$0.40-0.80/hr. Total project
  spend lands around $120-200.
- **Alternative:** [Lambda Labs](https://lambdalabs.com/) at ~$1.10/hr for
  A100 -- pricier but fewer preemptions for long runs.
- **After signup:** add $20 of credit. Generate an SSH key on your laptop
  (`ssh-keygen -t ed25519`) and paste the public key into
  Settings -> SSH Public Keys.

## Optional / later

### Pexels API — for the automated clip-downloader
- **Signup:** https://www.pexels.com/api/new/ (free, 30 seconds)
- **What it gives us:** authenticated access to Pexels video metadata + direct
  .mp4 URLs. `scripts/fetch_clip.py` uses this to bypass Cloudflare's anti-bot
  challenge that yt-dlp's generic extractor hits.
- **Free tier:** 200 requests/hour, 20,000/month. Way more than we need.
- **After signup:** copy the key and add `PEXELS_API_KEY=...` to `.env`.

### NVIDIA Developer account — only if we add a VLM-as-policy head later
- **Signup:** https://developer.nvidia.com/login (free)
- Needed for Isaac Lab. Out of scope for v1.

## After accounts exist — push the repo

Once the empty GitHub repo at
https://github.com/jeffrah00/robot-pet-cat exists, run:

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

The `scripts/init_git.ps1` / `scripts/init_git.sh` helpers do all of the
above; they'll print the remote-add command after the first commit.

## Local secrets

Copy `.env.example` to `.env` (already gitignored) and fill in the keys as
you create each account:

```bash
cp .env.example .env
# then edit .env
```
