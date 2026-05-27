import sys,numpy as np,mujoco
sys.path.insert(0,'src')
from robot_pet_cat.brain.env import BrainEnv,BrainEnvConfig
from pathlib import Path
from robot_pet_cat.brain.go2_policy import load_get_up_policy

cfg=BrainEnvConfig(use_physics_cat=True,walker_policy_path=Path('models/go1_walker_v0.pt'),max_steps_per_episode=600)
env=BrainEnv(cfg)
cat=env.cat
cat.get_up_policy=load_get_up_policy('models/get_up_go1_v1.pt')
cat._last_active_policy="walker"
mj_model,mj_data=env.mj_model,env.mj_data

PR=np.array([.3,1.6,-2.7,-.3,1.6,-2.7,.3,1.6,-2.7,-.3,1.6,-2.7],np.float32)
P=(3,4,5,0,1,2,9,10,11,6,7,8)

def reset():
    mujoco.mj_resetData(mj_model,mj_data)
    mj_data.qpos[cat._base_qpos_adr:cat._base_qpos_adr+7]=[0,0,.06,1,0,0,0]
    for i in range(12):
        mj_data.qpos[cat._joint_qpos_adr[i]]=PR[i]
        mj_data.qvel[cat._joint_qvel_adr[i]]=0
        mj_data.ctrl[cat._actuator_ids[i]]=PR[i]
    mujoco.mj_forward(mj_model,mj_data)

reset()
# Benchmark obs (op=0, no permutation):
xm=np.asarray(mj_data.xmat[cat._base_body_id]).reshape(3,3)
gv=-xm[2,:]
av=mj_data.sensordata[cat._gyro_adr:cat._gyro_adr+3]
la=np.zeros(12,np.float32)
jp_bench=np.array([mj_data.qpos[cat._joint_qpos_adr[i]] for i in range(12)],np.float32)
jv_bench=np.array([mj_data.qvel[cat._joint_qvel_adr[i]] for i in range(12)],np.float32)
obs_bench=np.concatenate([av,gv,jp_bench,jv_bench,la])
print(f"bench obs[:6]={obs_bench[:6]}")
print(f"bench jp={jp_bench}")

# PhysicsCat obs:
cat._last_action=np.zeros(12,np.float32)
obs_cat=cat._build_go1_getup_observation(mj_data)
print(f"cat   obs[:6]={obs_cat[:6]}")
print(f"cat   jp={obs_cat[6:18]}")
print(f"obs match: {np.allclose(obs_bench,obs_cat,atol=1e-4)}")
print(f"jp match:  {np.allclose(jp_bench,obs_cat[6:18],atol=1e-4)}")
print(f"bench jp[0]={jp_bench[0]:.4f}  cat jp[0]={obs_cat[6]:.4f}")
