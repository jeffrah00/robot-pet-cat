#!/usr/bin/env python3
"""Probe the get_up 13k checkpoint to extract its terminal sit-pose joint angles.

Mirrors play.py's env+policy construction, runs a headless rollout, and prints
the final joint_pos + root state. Outputs lines starting with SIT_HIND_POSE_PROBE.
"""

from __future__ import annotations
import os
import sys
import torch

sys.path.insert(0, '/workspace/unitree_rl_mjlab')
# Trigger task registration via import_packages side-effect.
import src.tasks  # noqa: F401

from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

CKPT = "/workspace/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-05-21_21-38-34/model_13000.pt"
TASK = "Unitree-Go2-GetUp"
N_STEPS = 300
N_PROBES = 3


def main():
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(TASK, play=True)
    agent_cfg = load_rl_cfg(TASK)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    robot = env.unwrapped.scene["robot"]
    all_poses = []
    for probe in range(N_PROBES):
        env.reset()
        obs, _ = env.get_observations()
        for step in range(N_STEPS):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        jp = robot.data.joint_pos[0].cpu().numpy().tolist()
        rp = robot.data.root_link_pos_w[0].cpu().numpy().tolist()
        rq = robot.data.root_link_quat_w[0].cpu().numpy().tolist()
        pg = robot.data.projected_gravity_b[0].cpu().numpy().tolist()
        all_poses.append((jp, rp, rq, pg))
        print("SIT_HIND_POSE_PROBE probe=%d joint_pos=%s" % (probe, jp))
        print("SIT_HIND_POSE_PROBE probe=%d root_xyz=%s" % (probe, rp))
        print("SIT_HIND_POSE_PROBE probe=%d root_quat=%s" % (probe, rq))
        print("SIT_HIND_POSE_PROBE probe=%d proj_grav=%s" % (probe, pg))
        print("SIT_HIND_POSE_PROBE probe=%d root_z=%.3f" % (probe, rp[2]))

    import statistics
    median_jp = [statistics.median(p[0][i] for p in all_poses) for i in range(12)]
    median_rz = statistics.median(p[1][2] for p in all_poses)
    print()
    print("SIT_HIND_POSE_PROBE median_joint_pos = %s" % median_jp)
    print("SIT_HIND_POSE_PROBE median_root_z    = %.4f" % median_rz)

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
