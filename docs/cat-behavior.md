# Why cat-like behavior emerges from this design

This doc exists because the obvious approach — "write a state machine that picks
between `sit`, `walk`, `jump`, ..." — produces something that feels like a robot
pet, not a cat. The architecture in `docs/architecture.md` is the answer to
"how do you get cat-feel instead of robot-feel without scripting?"

## What makes a real cat feel like a cat

Three ingredients, none of them about gait realism:

1. **Motion that doesn't look servo-driven.** Real cats shift weight before
   they move, anticipate landings, settle into a sit slowly, recover from
   small perturbations smoothly. Stiff joint-target trajectories feel
   uncanny.

2. **Decisions that surprise you in a coherent way.** You cannot predict
   when a cat will jump down from the couch. You *can* explain why it did
   once it does (saw something move, got bored, the sun moved). This is the
   feel of curiosity-driven behavior plus mild stochasticity, not the feel
   of a scheduler.

3. **An internal state that drifts.** Cats are sleepy, then alert, then
   playful. Same cat, same room, different behavior at 9am vs 9pm. Without
   this, behavior loops feel canned.

## How the three tiers map onto those three ingredients

| Cat ingredient | Implemented by |
|----------------|----------------|
| Servo-free motion | AMP style prior in Tier 1 (motion/) |
| Coherent surprise | Curiosity-driven RL + sampling temperature in Tier 3 (brain/) |
| Drifting internal state | Mood latent (brain/mood.py) |

You can think of Tier 1 as "how it moves," Tier 3 as "what it wants to do,"
and the mood latent as "what mood it's in right now." Tier 2 (skills/) is
purely a layer of competence — it doesn't carry character.

## Why this works without scripting any specific behavior

The three concrete behaviors you cared about:

**"Sits by the window."** No code says "go to window." Instead the comfort
reward is high near `window` + `elevated` + `warm` surfaces with low body
acceleration. The brain learns through RL that staying near windows produces
high reward when sleepy mood is dominant. Different windows in different
rooms work without modification.

**"Swats a ball."** No code says "hit ball." The play reward is proportional
to nearby small-object velocity, with a causal credit bonus when the cat
caused that velocity. When the play-weight is high (alert/playful mood), the
brain learns that the `swat` skill near a `play_target` object is a reward
hot spot. Add a feather toy — it'll swat that too.

**"Jumps on and off the couch."** The couch carries `elevated` + `soft`
flags, contributing to comfort. The cat learns to jump up to access that
reward when relaxed. When mood shifts to alert, comfort weight drops and the
cat jumps off to chase the ball.

None of these are scripted. All of them emerge from the reward structure +
the mood-modulated weighting.

## Why we don't use a VLA/VLM as the brain (yet)

A fine-tuned VLM as the brain is tempting: it could take language goals
("come here"), it could reason about novel objects, it would be cute. We're
deferring it for three reasons:

1. **Cat charm doesn't need language reasoning.** A cat doesn't take
   instructions. The autonomy is the point.
2. **RL with three intrinsic rewards plus mood is enough for cat-vibe.**
   Adding a VLM is real work and doesn't change the result much in the
   target scenarios.
3. **It's a clean upgrade path later.** Replace `brain/policy.py` with a
   VLM that emits the same `(skill_id, target)` action space. Everything
   else stays. Worth doing in Phase 7 once the core works.

## What can go wrong, and how we'll know

This architecture has predictable failure modes. Watching for these is most of
the Phase 5 work:

- **Curiosity-only collapse.** Cat runs around forever, never sits. Fix:
  increase comfort weight floor; tune mood drift toward sleepy.
- **Comfort-only collapse.** Cat sits in one spot forever. Fix: tighten
  curiosity-reward novelty decay; cap maximum dwell time per skill.
- **Skill thrashing.** Brain switches skills every step. Fix: penalize
  skill changes in reward; tune `decision_period_s` upward; reduce sampling
  temperature.
- **Mood that flickers.** OU noise is too high. Fix: lower `noise_sigma`,
  raise `decay_per_minute` so the latent has inertia.

## References used in designing this

- [AMP (Peng 2021)](https://xbpeng.github.io/projects/AMP/) — the style prior
  that produces cat-feel motion without scripting joint trajectories.
- [ICM curiosity (Pathak 2017)](https://arxiv.org/abs/1705.05363) — the
  exploration drive that produces coherent surprise.
- [Burda et al. 2018](https://arxiv.org/abs/1808.04355) — empirical guidance
  on tuning curiosity rewards.
- [Atanassov et al. 2024](https://arxiv.org/abs/2401.16337) — curriculum
  recipe for the jump skill specifically.

The combination is new for cats specifically, but each ingredient is
well-validated separately. We're composing, not inventing.
