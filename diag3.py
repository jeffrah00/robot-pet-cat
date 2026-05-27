import sys,numpy as np,mujoco
sys.path.insert(0,'src')
from robot_pet_cat.brain.env import BrainEnv,BrainEnvConfig
from pathlib import Path
from robot_pet_cat.brain.go2_policy import load_get_up_policy
from robot_pet_cat.brain.physics_cat import LocomotionCommand

cfg=BrainEnvConfig(use_physics_cat=True,walker_policy_path=Path('models/go1_walker_v0.pt'),max_steps_per_episode=600)
env=BrainEnv(cfg)
cat=env.cat
cat.get_up_policy=load_get_up_policy('models/get_up_go1_v1.pt')
cat._last_active_policy="walker"
mj_model,mj_data=env.mj_model,env.mj_data
trunk_id=mujoco.mj_name2id(mj_model,mujoco.mjtObj.mjOBJ_BODY,'trunk')
print(f"trunk_id={trunk_id}, _base_body_id={cat._base_body_id}")
PR=np.array([.3,1.6,-2.7,-.3,1.6,-2.7,.3,1.6,-2.7,-.3,1.6,-2.7],np.float32)
def reset():
    mujoco.mj_resetData(mj_model,mj_data)
    mj_data.qpos[cat._base_qpos_adr:cat._base_qpos_adr+7]=[0,0,.06,1,0,0,0]
    for i in range(12):
        mj_data.qpos[cat._joint_qpos_adr[i]]=PR[i];mj_data.qvel[cat._joint_qvel_adr[i]]=0;mj_data.ctrl[cat._actuator_ids[i]]=PR[i]
    mujoco.mj_forward(mj_model,mj_data)
    cat._last_active_policy="walker";cat._last_action=np.zeros(12,np.float32)
reset()
cmd=LocomotionCommand(vx=0,vy=0,yaw_rate=0,gait="get_up")
# Print comparison at first few ticks
for tick in range(60):
    cat.step(mj_data,cmd,0.05)
    qz=float(mj_data.qpos[2])
    xz=float(mj_data.xpos[trunk_id][2])
    if tick<5 or (tick>=20 and tick<25) or tick==50:
        print(f"tick={tick:3d}  qpos[2]={qz:.4f}  xpos[trunk][2]={xz:.4f}")
