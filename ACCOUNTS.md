# Accounts to create

Checklist of every external service this project depends on. Create them in order;
each step lists what it gives you and the direct signup URL.

## Required

### 1. GitHub — code hosting
- **You already have this.** Username needed for the remote.
- **Signup:** https://github.com/signup (skip)
- **What it gives us:** the canonical repo, Actions CI (2,000 free minutes/mo on
  private repos, unlimited on public).
- **Next step:** create an empty repo named `robot-pet-cat` on GitHub, then run the
  `git remote add` + `git push` commands at the bottom of this file.

### 2. Hugging Face Hub — models + datasets
- **Signup:** https://huggingface.co/join (free)
- **What it gives us:** versioned storage for the locomotion checkpoint, the
  fine-tuned VLA, and the cat-behavior dataset. Free tier is plenty for public
  artifacts; HF Pro ($9/mo) only if we want private datasets.
- **After signup:**
  1. Create an access token at https://huggingface.co/settings/tokens (Write scope)
  2. Save it; you'll run `huggingface-cli login` and paste it.

### 3. Weights & Biases — experiment tracking
- **Signup:** https://wandb.ai/site (free for personal use)
- **What it gives us:** training curves, hyperparameter sweeps, video logging of
  rollouts. The free "Personal" tier is unlimited for solo use.
- **After signup:** copy your API key from https://wandb.ai/authorize, you'll run
  `wandb login` and paste it.

### 4. RunPod — GPU compute (recommended primary)
- **Signup:** https://www.runpod.io/console/signup
- **What it gives us:** on-demand and spot A100/H100 instances, ~$0.40–0.80/hr for
  A100 spot. Better than AWS/GCP for sporadic ML work.
- **Alternative:** [Lambda Labs](https://lambdalabs.com/) — more reliable for
  long-running training, but pricier (~$1.10/hr for A100). Pick one to start; you
  can add the other later.
- **After signup:** add $20 of credit. Set up an SSH key on your laptop and add the
  public key under Settings → SSH Public Keys.

### 5. Cloudflare R2 (or Backblaze B2) — raw video object storage
- **Cloudflare R2 signup:** https://dash.cloudflare.com/sign-up
  - $0.015/GB/mo, **no egress fees** (huge deal for video)
  - 10 GB free
- **Backblaze B2 alternative:** https://www.backblaze.com/sign-up/cloud-storage
  - $0.006/GB/mo, modest egress fees
  - 10 GB free
- **What it gives us:** somewhere to keep the raw scraped cat videos (tens of GB)
  cheaply. Processed clips + features go to Hugging Face.
- **After signup:** create a bucket called `robot-pet-cat-raw`. Generate an access
  key + secret; save them to a `.env` file (already gitignored).

## Optional / later

### Anthropic API or OpenAI API — for behavior labeling assistance
- Useful in Phase 3 if we want to auto-caption short clips to bootstrap labels.
- Skip until Phase 3.

### NVIDIA Developer account — for Isaac Sim if we ever need photoreal training
- **Signup:** https://developer.nvidia.com/login
- Free, but Isaac Sim wants a powerful GPU. Skip unless we hit visual-fidelity
  limits with MuJoCo.

### YouTube Data API — if we go beyond yt-dlp for discovery
- **Signup:** https://console.cloud.google.com/ (Google Cloud), enable
  YouTube Data API v3
- Free quota: 10,000 units/day, enough for search-based discovery.
- Only needed if Pexels + curated channel lists aren't enough.

## After accounts exist — push the repo

Once you've created an **empty** GitHub repo at
`https://github.com/<your-username>/robot-pet-cat` (no README, no .gitignore — the
scaffold already has those), run:

```bash
cd "Robot pet cat"
git remote add origin git@github.com:<your-username>/robot-pet-cat.git
git branch -M main
git push -u origin main
```

If you prefer HTTPS over SSH:
```bash
git remote add origin https://github.com/<your-username>/robot-pet-cat.git
```

Tell me your GitHub username when you're ready and I can pre-fill the remote URL.

## Local secrets

A `.env.example` file is included; copy it to `.env` (already gitignored) and fill in
the keys as you create each account:

```bash
cp .env.example .env
# then edit .env
```
