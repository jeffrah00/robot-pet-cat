"""Mid-level skills — subgoal-conditioned policies trained under the AMP style prior.

Each skill is a small RL policy specialized for one motor task. They share the
cat motion style enforced by `motion/`. The brain module chooses which skill to
invoke based on visual context and mood.

Initial skill set:
  - walk_to(target_xy)          — locomotion with goal-relative velocity command
  - sit                         — lower haunches, stand still
  - lie_down                    — fold legs, rest belly
  - stretch                     — long forward extension
  - groom                       — head-to-paw oscillation
  - swat(object_xyz)            — approach + front-paw lift + nudge
  - jump_to(surface_xyz)        — preparatory crouch + jump + landing
  - crouch                      — pre-pounce low pose
  - look_at(point_xyz)          — neck/head pose toward a target

We deliberately have fewer skills than a real cat. The brain composes them.
"""
