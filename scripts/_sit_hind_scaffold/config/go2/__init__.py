"""Unitree Go2 crouch task registration."""

from mjlab.tasks.registry import register_mjlab_task

# Reuse the velocity OnPolicyRunner — it's task-agnostic.
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_go2_sit_hind_env_cfg
from .rl_cfg import unitree_go2_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-Go2-Sit-Hind",
  env_cfg=unitree_go2_sit_hind_env_cfg(),
  play_env_cfg=unitree_go2_sit_hind_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
