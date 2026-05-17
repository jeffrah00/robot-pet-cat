# Cat motion corpus

The AMP discriminator is only as good as the reference distribution it
sees. One walking clip teaches "this exact 20-second gait cycle"; a
behaviorally diverse corpus teaches "this is what cat-shaped motion
distributions look like across activities."

This document is the curation framework: what behaviors we want, how
many clips per behavior, and the per-clip metadata schema. The actual
clip list lives in `data/motion_clips_raw/manifest.yaml`.

## Behaviors

A real cat's motion vocabulary breaks roughly into six clusters. Target
counts are rough — more is fine, but discriminator overfits to
duplicates so dedupe across similar clips.

### 1. Locomotion (4–6 clips)
Walking different speeds, trotting, galloping, turning in place. Most of
the policy's runtime budget is spent moving, so this needs the most
variation in stride length, cadence, and ground type.

- walk_slow_indoor (~3-6s, low velocity)
- walk_normal_outdoor (~5-10s)
- trot (~3-5s)
- gallop / run (~2-4s, may be hardest to retarget cleanly)
- turn_in_place (~3-5s)

### 2. Resting and lounging (3–5 clips)
Static or near-static poses the policy should be able to settle into.
Without these the policy never learns that "stop and lie down" is
cat-shaped.

- lying_side (~5-10s, ribs visible)
- lying_curled (~5-10s, head tucked)
- lying_belly_up (~3-5s, optional)
- sitting_upright (~5-10s, alert pose)
- loaf (sitting with paws tucked) (~5-10s)

### 3. Transitions (3–5 clips)
The hardest poses to ad-lib from steady states. Worth having explicit
clips so the discriminator rewards them rather than punishing as
"weird in-between."

- stand_to_sit
- sit_to_stand
- sit_to_lying
- lying_to_sit_or_stand
- crouch_to_pounce

### 4. Grooming (2–4 clips)
Self-care poses. Jeff specifically called out leg-licking; the robot
has no mouth but the *body shape* (curled spine, leg lifted toward
where the head is) is what AMP cares about.

- lick_paw
- lick_leg_or_belly (foreleg or hindleg raised toward midline)
- scratch_head_with_hind_leg
- shake_off (full-body shake)

### 5. Stretching (2–3 clips)
High-amplitude poses that exercise joint extremes. These teach the
policy that big stretches are part of the natural distribution.

- arch_back_yoga (Halloween-cat pose, downward dog inverted)
- forward_stretch (front paws extended forward, low haunches)
- side_stretch (lying-down full leg extension)

### 6. Play and predatory (2–4 clips)
The "alive" behaviors. Tier 2 skill training will probably need its own
versions of these too, but having them in Tier 1 keeps the base policy
from being a boring locomotion-only robot.

- crouch_stalk (low, slow forward creep)
- wiggle_butt (pre-pounce side-to-side)
- pounce / pounce_attack
- swat (single-paw)
- jump_up / jump_down (vertical)

**Total target**: 16–27 clips covering all six clusters. Even half this
is a big upgrade over the single walk clip we have now.

## Per-clip schema

`data/motion_clips_raw/manifest.yaml` is a list of entries:

```yaml
clips:
  - id: walk_slow_01                         # filesystem-safe identifier
    behavior: walk_slow                      # cluster tag from above
    source: pexels                           # pexels | youtube | other
    source_id_or_url: "12345678"             # Pexels ID or full URL
    license: pexels_free                     # whatever covers it
    crop:                                    # optional time window
      start_s: 1.5
      end_s: 8.0
    notes: "side-view, gray short-haired, outdoor pavement"
```

Required: `id`, `behavior`, `source`, `source_id_or_url`. Everything
else is optional metadata.

## Why YAML manifest + scripts (not just raw downloads)

We want the curation step (which clips to include) decoupled from the
download mechanism (Pexels API vs yt-dlp) and from the retargeting
pipeline (DLC + our retargeter). YAML makes the curation reviewable in
git diff; scripts make adding a new clip a single-line manifest edit.

## Pipeline

```
manifest.yaml
   |
   v
scripts/fetch_clips.py        --> data/motion_clips_raw/<id>.mp4
   |
   v
scripts/retarget_all.py
   (DLC pose extraction)      --> data/motion_clips_raw/<id>_keypoints.json
   (our retargeter)           --> data/motion_clips/<id>.npz
   |
   v
ReferenceBuffer.from_dir(data/motion_clips/)
   |
   v
AMP discriminator pre-training + per-iter updates
```

Failures at any stage drop the clip with a message; the buffer just
includes whatever made it through. Aim for >= 80% retargeting success
rate on the curated set; if it's lower, the manifest needs more
side-view filtering.

## License hygiene

Pexels content is free to use under their license but we should still
record the source ID per clip so attribution is possible later. YouTube
clips need per-channel checking; default assumption is "fair-use
research, do not redistribute the raw videos in our repo." `.gitignore`
already excludes `data/motion_clips_raw/*.mp4`.
