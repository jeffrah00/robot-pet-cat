#!/bin/bash
source /workspace/robot-pet-cat/.env
source /workspace/mjlab_venv/bin/activate
cd /workspace/robot-pet-cat
echo "[$(date)] RENDER3 START" > renders/render3_log.txt
MUJOCO_GL=osmesa python scripts/render_brain_3d.py --random --steps 6000 --checkpoint nonexistent --out renders/brain3d_random_5min.mp4 >> renders/render3_log.txt 2>&1
echo "[$(date)] RENDER3 DONE" >> renders/render3_log.txt
