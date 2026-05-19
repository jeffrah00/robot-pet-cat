# Brain design v2 — character-AI framing, not pure RL

## What this doc is

This is a revision of the original Tier 3 brain design (documented in
`architecture.md` and `cat-behavior.md`) based on a 2026-05-19 design
conversation. The original design is the *substrate*; nothing here breaks it.
What changes is the architecture layered on top.

The premise of v2: cat-ness is not a control problem, it's a character problem.
Treat the brain as character-AI for an embodied agent rather than as an RL
policy that picks skills every step. The motor cortex (Tier 1) and skill
library (Tier 2) are the body and motor vocabulary; the brain is the
*character* that lives in that body.

## Why v2

The original design was a clean canonical RL setup: PPO over a composite
intrinsic reward (curiosity + comfort + play), with mood drift biasing the
reward weights. The argument in `cat-behavior.md` is that this composition is
sufficient for cat-feel.

Two problems became visible after the AMP failures and the research-mode
conversation:

1. **The robotics literature has solved the motor-cortex problem** (APEX, Lifelike
   Agility, etc.), so trying to invent novel ideas at Tier 1 wastes effort.
   The novel territory is the brain.
2. **Per-step skill selection is not how cats actually behave.** Cats enter
   *modes* (resting, stalking, playing) that persist for tens of seconds.
   A policy that picks a skill every 2 s, even with a mood prior, samples too
   quickly and produces busy behavior instead of cat behavior.

The v2 design addresses both: invest the design budget at Tier 3, and structure
Tier 3 around persistent behavioral modes rather than per-step skill choice.

## Five architectural additions

Each is additive to the original scaffold. None require deleting existing code.

### 1. Behavioral attractors as a layer between mood and policy

A small set (~4-8) of high-level *modes* — `resting`, `observing`, `stalking`,
`playing`, `grooming`, `exploring` — each one a coherent set of permissible
skills and a coherent target time scale (seconds to a minute). The brain
operates at two frequencies:

- **Mode transitions** happen rarely (every 10-60 s), gated by mood + scene
  saliency. Concretely: a low-frequency switching policy with a transition
  matrix biased by the current mood vector.
- **Within-mode skill selection** happens at the existing 0.3-1 Hz tick, but
  only over the skills the current mode allows. `resting` mode can pick
  among {`sit`, `lie_down`, `look_at`}; `stalking` among {`crouch`, `walk_to`,
  `look_at`}; `playing` among {`swat`, `walk_to`, `jump_to`}; etc.

This makes "the cat is in stalking mode" a first-class architectural element.
It also dramatically reduces the search space the per-step policy explores,
which should accelerate training and reduce skill thrashing (one of the named
failure modes in `cat-behavior.md`).

Implementation sketch: add `src/robot_pet_cat/brain/attractor.py` with:
- `Attractor` enum / dataclass listing the modes
- `mode_to_skills` mapping
- `ModePolicy` — a small categorical over modes, transitions sampled at
  `mode_decision_period_s` from a softmax over (mood × scene_saliency
  features)

The existing `brain.policy.BrainPolicy` then dispatches over
`current_mode.allowed_skills` instead of the full skill set.

### 2. Gaze-first split

In nature, gaze leads body. A cat fixates on a target several seconds before
deciding whether to approach. Almost all robotics policies drive the head as a
consequence of body motion; the v2 design inverts this.

Two coordinated policies inside Tier 3:

- **Gaze policy** at ~5-10 Hz: picks `look_at(point_xyz)` based on a
  saliency model over scene observations (novel objects, recent motion,
  contrast). Outputs are head_yaw / head_pitch commands consumed by the
  motor controller.
- **Body policy** at the existing 0.3-1 Hz: picks skills, but receives the
  current gaze target as an additional observation. "Approach what I'm
  looking at" becomes a natural action; "ignore what I'm looking at"
  becomes a natural action; "look at something then decide" becomes the
  default temporal pattern.

The two policies can share a backbone but have separate heads. Training-wise,
the gaze policy is cheaper (small action space, dense saliency signal) and
should learn quickly; the body policy gets the gaze trace as a stable feature.

Implementation sketch: `src/robot_pet_cat/brain/gaze.py` with a saliency
network (input: scene_state, output: per-entity saliency score → softmax →
sampled target) and a fast tick loop that runs alongside the body policy.

### 3. Pause-as-default action prior

Standard robotics rewards penalize stationarity. Cats reward it: cats are still
~80% of the time. The v2 design inverts the prior.

Concretely: add a "hold" action to the body policy's action space alongside the
real skills. Reward a small positive bonus for holding when no salient stimulus
is present in the scene. This is *not* a passive `sit` skill — `hold` means
"continue whatever pose the previous skill ended in, do nothing new." The
no-op is the default; doing things is the exception.

Operationally:
- Body policy action space: `Discrete(n_skills + 1)`, with `hold` at index 0.
- Hold-bonus reward: `+0.01 per tick` when `scene_saliency < threshold` and
  `cat_state == "at_rest"`.
- This pairs with the comfort reward: hold + on-soft-surface compounds.

Effect: the brain learns to do nothing unless the world asks for something.
This is the single most "cat" architectural choice in v2.

### 4. Expanded cat-specific intrinsic reward library

The original three rewards (curiosity, comfort, play) cover the big strokes
but miss several characteristically-feline drives. v2 adds:

- **Vantage-seeking** — bonus for being at higher z than typical (perched on
  elevated surfaces, looking down). Different from `comfort.elevated`: vantage
  is about *altitude*, comfort is about *softness*.
- **Warmth-seeking** — already partially covered by `comfort.warm`, but
  promote to a first-class drive with its own weight and a temperature-aware
  scaling (if we ever simulate temperature; for v0 it's static patches).
- **Ambush-position seeking** — bonus when the cat is near an `elevated` body
  and has clear line of sight to a `play_target` (precondition for pouncing).
  Cats spend time *staging*, not just *acting*.
- **Prey-tracking** — bonus for keeping gaze fixated on a moving object,
  scaled by object speed. Couples directly with the gaze policy.
- **Social-distance maintenance** — bonus for being neither too close nor too
  far from a tagged `human` entity (when one is in the scene). Cats prefer a
  middle distance.

All five extend `brain/rewards.py` as additional dataclasses with `compute()`
methods. The composite reward becomes a 6-8 term weighted sum (still
mood-modulated weights), not 3.

### 5. Animation principles as motor primitives

Animators have a vocabulary for "what makes motion look alive" that robotics
mostly ignores: anticipation, follow-through, staging, squash-and-stretch,
slow-in/slow-out. These can slot into Tier 2 / Tier 3 as small additions.

Most relevant for cats:

- **Anticipation**: before a forward skill (`walk_to`, `swat`, `jump_to`),
  emit a brief backward/downward weight-shift motion. ~200 ms. Implemented
  as a pre-roll commanded by the brain when transitioning into the skill.
- **Follow-through**: after `swat` or `jump_to`, hold the final pose briefly
  rather than snapping back to neutral. Implemented as a post-roll.
- **Staging**: face the action target before acting. Already partly covered
  by `look_at` + the gaze-first architecture; formalize as a "orient first"
  pre-step.

These are not learned. They're small scripted modifiers around skill
invocations. They're disproportionately responsible for "this thing is alive"
perception.

## How v2 sits on top of v1

The v1 scaffold stays intact:

| v1 file | v2 status |
|---|---|
| `brain/mood.py` | **unchanged** — mood latent is correct, drives v2 weights too |
| `brain/rewards.py` | **extended** — original 3 reward classes stay; add 5 new ones |
| `brain/policy.py` | **refactored** — `BrainPolicy` becomes `BodyPolicy`, gains hold action; add `ModePolicy` + `GazePolicy` siblings |
| `docs/architecture.md` | **still mostly accurate** — Tier 1 / Tier 2 unchanged, Tier 3 needs an addendum block |
| `docs/cat-behavior.md` | **still mostly accurate** — the "why" arguments still hold; failure-mode catalog still relevant |

New files v2 introduces:

| New file | Purpose |
|---|---|
| `brain/attractor.py` | mode definitions + mode-transition policy |
| `brain/gaze.py` | saliency model + fast-tick gaze policy |
| `brain/animation.py` | anticipation / follow-through / staging modifiers |
| `scene/scene_state.py` | already written 2026-05-19 — flag-based scene state extractor |

## Open questions

These are knowingly deferred to subsequent sessions:

1. **Mode count and granularity.** ~6 modes feels right, but the actual set
   needs iteration. Should `grooming` be its own mode or a within-`resting`
   skill? Empirically determine.
2. **Mode-transition frequency.** 10-60 s feels right. Too fast and we're
   back to skill thrashing at the mode level; too slow and the cat looks
   stuck.
3. **Gaze policy architecture.** Top-K saliency softmax is the obvious
   starting point. A more elaborate attention model (transformer over scene
   entities) is overkill for the v0 scene with 3 objects but might matter at
   Phase 4 with more clutter.
4. **Hold-bonus calibration.** If the bonus is too high the cat never moves;
   too low and it has no effect. Need empirical tuning, probably against a
   target "fraction of time spent holding" hyperparameter (~70-85% feels
   cat-like).
5. **How animation principles compose with learned skills.** Anticipation +
   follow-through wrap skills; do they wrap individual skill calls, or do
   they live in a "skill-with-anim-decorator" wrapper? Probably the latter.
6. **Whether to train mode policy and body policy jointly or sequentially.**
   Joint is principled (option-critic style) but harder; sequential
   (pre-fix modes by hand, then train within-mode skill policies) is easier
   and may be enough.

## What v2 *doesn't* claim

- It doesn't claim novel motor control (we use the existing Go2 walker / APEX).
- It doesn't claim novel reward shaping in isolation (the individual rewards
  are standard).
- It doesn't claim novel hierarchical RL (options framework exists since
  Sutton 1999).
- It *does* claim that the specific composition — behavioral attractors +
  gaze-first + pause-default + cat-specific intrinsic reward library +
  animation principles, all wired together for embodied character behavior
  on a real motor controller — is not in any single published paper we found
  during the 2026-05-19 literature search. The combination is the work.

## Implementation order (suggested)

These can be done independently and in any order. Listed roughly by
information-per-hour:

1. **Hold action + pause-default reward** — smallest code change, biggest
   behavioral effect. Just add a `hold` action and one new reward term.
2. **Behavioral attractors** — `attractor.py` with ~50 lines and a
   `mode_to_skills` table; modify `BodyPolicy` to dispatch within the active
   mode's skill set.
3. **Expanded intrinsic reward library** — add the 5 new dataclasses in
   `rewards.py` with stubs, then implement one at a time.
4. **Gaze-first split** — `gaze.py` with a small saliency model. This is the
   biggest delta because it adds a new policy + new training signal; do it
   only after the simpler additions look right.
5. **Animation principles** — wrap skill invocations after the brain is
   working end-to-end.

## References

- `docs/architecture.md` — original three-tier design
- `docs/cat-behavior.md` — design rationale for why three tiers produce
  cat-feel
- Memory file `play_py_never_exits_headless.md` — operational gotcha
- Memory file `amp_imitation_track_failed.md` — what we learned from
  earlier AMP attempts
- APEX paper (Wang et al. 2025, arXiv 2505.10022) — motor cortex
  recommendation
- Lifelike Agility (Zhang et al. 2024, Nature MI) — closest published peer
  for the three-tier idea
- Options framework (Sutton, Precup, Singh 1999) — original hierarchical RL
  framing the attractor layer reuses
