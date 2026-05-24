# Walker-MoB Porting Plan

**Status:** Proposal — 2026-05-25
**Author:** Claude (with Jeff)
**Goal:** One skill that actually works, built by extending the working walker's command space rather than training a new from-scratch policy.

---

## TL;DR

Every from-scratch RL skill (crouch, get_up v1-v10, jump_to v3-v5) and every scripted velocity-command "skill" failed. The only thing that produces convincing motion is the original Unitree walker (`model_6400.pt`, `Unitree-Go2-Flat`). We've been retraining locomotion fifteen times when we should be teaching one policy more vocabulary.

The walk-these-ways Multi-of-Behaviors (MoB) design from Margolis & Agrawal (CoRL 2022) is the canonical example of this paradigm: a single command-conditioned policy where extra command dimensions (`body_height`, `pitch`, `footswing_height`, `gait_freq`, etc.) make the same network produce qualitatively different gaits. The Go2 port (Teddy-Liao) shows the design transfers to our robot. Our `unitree_rl_mjlab` stack already has many of the ingredients (a `phase` observation, a `foot_gait` reward, a `pose` reward, command sampling with curriculum).

**V1 proof-of-concept:** add a single new command dim, `body_height_cmd ∈ [-0.15, +0.10] m` (offset from default 0.30 m base height), with one new reward (`track_body_height`). That alone gives us creep-low / normal / stretch-tall behaviors from one policy. ~50 LOC of new code, warm-started from `model_6400.pt`, trained ~3000 iters on a single RunPod GPU.

If V1 works (success criterion: a single 20-second rollout where body height tracks a sweep from 0.20 → 0.30 → 0.40 m while velocity-tracking holds), V2 adds `pitch` and `footswing_height`. V3 adds full gait conditioning (freq/phase/offset) and we'd have a single policy covering ~7 of our 15 target behaviors. Genuinely-non-locomotor behaviors (lie_down_still, sit_still, get_up_from_lying) remain out of scope for this paradigm and need separate treatment.

---

## Why this paradigm

Three reasons it should work where the others didn't:

The first is sample efficiency. A from-scratch crouch policy has to relearn standing, balance, joint limits, contact reasoning, foot phasing, and *then* the actual crouch — six things at once with sparse reward. MoB hands the policy all six of those for free and asks it to learn only the seventh thing (height tracking). That's why the original walk-these-ways policy converges in ~10k iters covering five gait families plus all the body-shape dims, while our get_up v9 ran 10k iters and produced sphynx pose.

The second is that it matches the cat. A cat creeping low and a cat walking tall and a cat sitting low are the same locomotion controller at different postures — they're not separate motor skills. Treating them as separate RL policies was always a category error.

The third is that it preserves the architecture we already validated. The brain still chooses an attractor mode, modes still emit a command vector, the walker still consumes commands. Only the command vector grows from 3 dims to 4 (V1) → 6 (V2) → 10 (V3). The PhysicsCat bridge, the BrainEnv, the attractor mask, the ICM, the stochastic decision period — none of that changes.

---

## Reference architecture (walk-these-ways MoB)

From `Improbable-AI/walk-these-ways` (`go1_gym/envs/base/legged_robot.py`, 1694 lines):

**Command vector — 15 dims.** Indices 0-2 are standard velocity commands; 3 is `body_height_cmd` (offset from `base_height_target=0.30 m`, range `[-0.25, +0.15]`); 4 is `gait_frequency_cmd` ∈ [2.0, 4.0] Hz; 5-7 are gait phase/offset/bound for the four-foot scheduler; 8 is gait duration (fixed at 0.5); 9 is `footswing_height_cmd` ∈ [0.03, 0.35] m; 10-11 are pitch and roll commands; 12-13 are stance width and length; 14 is an auxiliary reward coefficient.

**MoB-specific reward terms.** Five terms do the heavy lifting:
- `tracking_contacts_shaped_force` (weight +4.0) penalizes ground force on a foot during commanded swing.
- `tracking_contacts_shaped_vel` (weight +4.0) penalizes foot velocity during commanded stance.
- `orientation_control` (weight -5.0) tracks commanded pitch/roll via projected gravity.
- `raibert_heuristic` (weight -10.0) penalizes foot-placement error from the Raibert offset.
- `feet_clearance_cmd_linear` (weight -30.0) tracks commanded swing height during swing phase.
- Plus `body_height` reward (weight +10.0) tying `commands[:,3]` to actual base height.

**The gait clock subsystem (`_step_contact_targets`, lines 830-906) is non-negotiable.** It derives per-foot desired-contact-states from the gait command dims via a von Mises smoothed phase, and feeds the two `tracking_contacts_shaped_*` rewards. Without it those rewards collapse to noise.

**Observation construction.** Policy obs = 70 dims: `ang_vel(3) + lin_vel(3) + grav(3) + commands*scale(15) + (dof_pos - default)(12) + dof_vel*0.05(12) + actions(12) + last_actions(12) + clock_inputs(4)`, wrapped in a 30-step `HistoryWrapper`. The `clock_inputs` are `sin(2π·foot_phase)` per foot — the policy literally sees the metronome.

**Training tricks that matter.** `randomize_lag_timesteps=True, lag_timesteps=6` (sim2real). `use_terminal_body_height=0.05` (prevents the policy from lying flat when `body_height_cmd=-0.25`). `only_positive_rewards_ji22_style=True, sigma_rew_neg=0.02` (keeps the negative MoB penalties from triggering early termination). `action_smoothness_1=-0.1, action_smoothness_2=-0.1` (the big footswing/clearance rewards produce jerky policies without these). Friction/restitution/mass/Kp/Kd domain randomization.

**Go2 port deltas (Teddy-Liao).** PD gains 20→25 / 0.5→0.6, thigh defaults asymmetric (front 0.8, rear 1.0), `flip_visual_attachments=True`. Critically: they reuse the Go1 actuator-net (no Go2 net trained). For our mjlab port this is moot — we use mjlab's MuJoCo PD directly, no actuator-net.

---

## What `unitree_rl_mjlab` already has

The agent that crawled the repo found:

- **Task family:** `velocity` (no task literally named `Unitree-Go2-Velocity`; the registered IDs are `Unitree-Go2-Flat` and `Unitree-Go2-Rough`).
- **Command term:** `UniformVelocityCommandCfg` (`src/tasks/velocity/mdp/velocity_command.py`), 3 dims (`lin_vel_x`, `lin_vel_y`, `ang_vel_z`), `vel_command_b: [N, 3]`.
- **Observation:** policy obs already includes `base_ang_vel + projected_gravity + command + phase + joint_pos + joint_vel + actions + height_scan`. The `phase` observation (`mdp.phase` with `period=0.6`) is essentially a fixed-trot version of walk-these-ways' clock_inputs.
- **Reward terms that overlap MoB:** `foot_gait` (gait phasing via period+offset), `foot_clearance` (fixed `target_height=0.10`), `pose` (variable_posture), `body_orientation_l2`, `track_linear_velocity`, `track_angular_velocity`. Plus standard penalties (`joint_acc_l2`, `action_rate_l2`, etc.).
- **Go2 specifics in `config/go2/env_cfgs.py`:** `foot_gait.offset = [0.0, 0.5, 0.5, 0.0]` (trot), illegal-contact termination, per-joint `pose` std dicts.
- **PPO config:** actor/critic `(512, 256, 128)` ELU, `entropy_coef=0.01`, adaptive LR, `desired_kl=0.01`, 10001 max iters.

**So mjlab already has roughly 60% of MoB.** The fixed-trot `phase` + `foot_gait` substitutes for walk-these-ways' clock_inputs + tracking_contacts_shaped (at the cost of fixed gait freq/offset). The `foot_clearance` reward is the seed of `feet_clearance_cmd_linear` (just needs to read a commanded height instead of a constant). What's *missing* is body-shape commands (height, pitch, footswing) and tracking rewards for them.

---

## V1 plan — body_height conditioning only

**Scope.** Add one command dim. Add one reward term. Keep everything else (gait, footswing, pitch) at the fixed defaults the current walker already produces. Warm-start from the working `model_6400.pt`. Train ~3000 iters. Verify with a single body-height sweep rollout.

**Why this scope.** It's the absolute smallest change that proves the paradigm. If body-height conditioning works, `pitch` and `footswing_height` are mechanical extensions of the same pattern. If body-height conditioning *doesn't* work, no amount of adding more dims will fix it — so we'd learn that quickly and pivot.

### File-by-file changes

All paths are repo-relative inside `unitree_rl_mjlab` on the RunPod side.

**1. `src/tasks/velocity/mdp/velocity_command.py` — extend the command term.**
- Add `body_height: tuple[float, float]` to `UniformVelocityCommandCfg.Ranges`.
- Grow `self.vel_command_b` from `[N, 3]` → `[N, 4]`. Last dim is `body_height_cmd` (offset from base_height_target).
- In `_resample_command`, fill index 3 from `Ranges.body_height` uniform sampling, with a configurable fraction (start at 0.0, ramp to 1.0 over training) of envs that get `body_height_cmd=0` for warm-start stability.

**2. `src/tasks/velocity/mdp/rewards.py` — add `track_body_height`.**
- Signature mirrors `track_linear_velocity`: `def track_body_height(env, command_name="twist", std=0.05) -> torch.Tensor`.
- Reads `vel_command_b[:, 3]` (the new dim) and the robot's actual base height from `env.scene["robot"].data.root_pos_w[:, 2]`.
- Returns `exp(-(actual - (base_height_target + commanded_offset))**2 / std**2)`.

**3. `src/tasks/velocity/velocity_env_cfg.py` — register the new reward.**
- Add `track_body_height: RewardTermCfg(func=mdp.track_body_height, weight=2.0, params={"std": 0.05})` to the `rewards` dict in `make_velocity_env_cfg`.
- Weight starts at 2.0 (matches `track_linear_velocity`). Tune later.

**4. `src/tasks/velocity/config/go2/env_cfgs.py` — set Go2 ranges.**
- Pass `body_height=(-0.10, +0.10)` for V1 (narrower than walk-these-ways' [-0.25, +0.15] — we don't yet have terminal_body_height termination on the low end).

**5. `src/tasks/velocity/config/go2/__init__.py` — register new task.**
- `register_mjlab_task(task_id="Unitree-Go2-Flat-MoB", env_cfg=..., play_env_cfg=..., rl_cfg=..., runner_cls=VelocityOnPolicyRunner)`.

**6. Optional V1.5: extend `foot_clearance` reward** to read the commanded swing height instead of a fixed `0.10 m`. Adds one command dim (`footswing_height`) for ~10 LOC. Defer if V1 alone works.

Total: ~80 lines of new code, no new files.

### Warm-start strategy

The walker checkpoint `model_6400.pt` has obs_dim = 47, action_dim = 12. After V1 changes, obs_dim = 48 (the command grew by 1). We need to load 47-dim weights into a 48-dim first linear layer.

Two clean options:

**(A) Pad-and-zero-init.** Load the state dict, take the actor's first linear weight `W: [hidden, 47]`, pad to `[hidden, 48]` with zeros in the new column. Same for the critic. This makes the new dim a no-op at episode 0 — the policy behaves identically to `model_6400.pt`. Then PPO training pulls signal into the new column as soon as `body_height_cmd` starts varying.

**(B) Command-curriculum.** Even with (A), keep `body_height_cmd=0` for the first ~500 iters by setting the sampling fraction of "zero-command" envs to 1.0. After 500 iters, ramp the fraction to 0 over the next 1000 iters. This ensures the early training stays close to the warm-start basin and avoids destroying the walker before the height-tracking reward starts contributing.

Recommend both. The `mjlab_train_resume_flags` memory ([[mjlab_train_resume_flags]]) covers the flags: `--agent.resume True --agent.load-run <model_6400_dir> --agent.load-checkpoint model_6400.pt`. We'll need a small Python helper to do the padding before the resume — call it `scripts/warm_start_to_mob.py`, ~30 LOC.

### Training schedule

| phase | iters | what changes |
|---|---|---|
| warm-up | 0-500 | all envs `body_height_cmd=0`, weight on `track_body_height` ramps 0→2.0 |
| widen | 500-1500 | command sampling ramps from 0% to 100% varying; weight stays 2.0 |
| converge | 1500-3000 | full command range, normal training |
| evaluate | — | render body-height sweep |

Run on a single RunPod GPU. Expect ~3 hours wall-clock based on the current Go2-Flat throughput (~1000 iters/hour with 4096 envs).

### Success criteria

Three checks, each a hard gate:

The first is the **body-height sweep render.** Script a `play.py` rollout that holds `lin_vel_x=0.3 m/s, ang_vel_z=0` and sweeps `body_height_cmd` from `-0.10` to `+0.10` to `0` over 20 seconds. Success = visibly different postures (low creep vs tall walk), measured base height tracks within ±2 cm of commanded, no falls, velocity tracking holds.

The second is a **multi-condition reward check.** Across 4096 parallel envs with random commands, mean `track_body_height` reward > 0.7 and mean `track_linear_velocity` reward stays > 0.7 (i.e., we didn't trade velocity tracking for height tracking).

The third is a **brain integration check.** Plug the MoB walker into `PhysicsCat`, set up two attractor modes — `creep` (sets `body_height_cmd=-0.10`) and `prowl` (sets `body_height_cmd=0`) — and verify a single brain rollout produces visibly different body heights when those modes activate. This validates the wiring story: that the existing attractor architecture trivially extends with new command dims.

---

## V2 and beyond (not now)

V2 adds `pitch` ∈ [-0.2, +0.2] rad and `footswing_height` ∈ [0.05, 0.20] m. Pitch unlocks "look up" / "head down stalking" postures via base-frame tilt. Footswing unlocks "tiptoe creep" (low swing) vs "high step" (cat trotting over an obstacle). Both add one command dim, one tracking reward, ~30 LOC each.

V3 adds full gait conditioning (`gait_freq, gait_phase, gait_offset`). This requires replacing the current fixed-period `phase` obs and `foot_gait` reward with the walk-these-ways gait-clock subsystem — the only V where we need the von Mises phase machinery. Bigger change (~200 LOC), and only worth doing if V2 proves out.

**Skills covered by V1-V3:** walk, creep, prowl, stalk, slow_walk, fast_walk, sit_low_walk, stretch_walk — roughly half of the 15.

**Skills NOT covered:** sit_still, lie_down, lie_down_still, get_up_from_lying, swat, groom, jump_to_couch. These are non-locomotor or transient and don't fit the MoB frame. Likely paths for them later:
- **Stillness skills (sit_still, lie_down_still)** — may emerge naturally from MoB at low body_height + zero velocity command, OR may need a separate scripted-pose policy with the walker disengaged.
- **Transient skills (get_up, jump_to)** — these are the genuine candidates for paradigm 1 (residual policy on top of MoB walker) or a small dedicated RL policy with the MoB walker as initial-state.
- **Manipulation-like skills (swat, groom)** — likely scripted keyframes on top of a "stationary" command from the walker; revisit after V1.

---

## Risks and open questions

**The biggest risk** is that body-height tracking conflicts with foot-clearance and gait-phase tracking inside the existing reward structure. mjlab's `foot_clearance` reward has a fixed `target_height=0.10`; if we lower body height to 0.20 m but the policy still tries to lift feet to 0.10 m, that's a 50% body height — feet won't actually swing. Mitigation: in V1.5 we should also widen `foot_clearance` to read commanded swing height (V1.5 is small enough to roll into V1 if needed).

**Second risk:** warm-start may not work cleanly. The walker's first linear layer was trained without a body_height channel; zero-init for the new column is correct in expectation but may sit in a poor PPO gradient region. Fallback: train from scratch with the wider command space. Loses the 6400 iters but is mechanically simple.

**Third risk:** the lower end of `body_height_cmd` may need `terminal_body_height` to prevent reward-hacking by lying down. walk-these-ways uses 0.05 m. For V1 with range `[-0.10, +0.10]`, lowest commanded height is ~0.20 m, so a terminal of 0.10 m is safe. Add as a termination term in `mdp/terminations.py`.

**Open question:** should we keep the existing `pose` (variable_posture) reward? It currently penalizes joints diverging from default pose. At low body_height that may fight against the necessary joint flexion. May need to reduce its weight or remove it when `|body_height_cmd|` is large. Decision deferred to first training run.

---

## What I'm asking

1. **Confirm V1 scope is right** — body_height only as the single new dim, not body_height + pitch + footswing all at once.
2. **Approve the warm-start approach** (pad-and-zero-init + command curriculum) vs train-from-scratch.
3. **Approve task naming** — `Unitree-Go2-Flat-MoB` and we add `-Rough-MoB` later, not first.
4. **Decision on `pose` reward** — keep as-is for V1 and see what breaks, or proactively reduce its weight?

Once those four are answered, the actual implementation is a one-day RunPod session: write the 80 LOC patch, push, kick off the 3-hour training run, render the sweep, evaluate. If V1 fails the criteria, we'll have a clean signal about whether to retry with different ranges/weights or pivot to paradigm 1 (residual).
