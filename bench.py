import sys,numpy as np,mujoco
sys.path.insert(0,'/workspace/robot-pet-cat/src')
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
def test(op,ap,steps=600):
    reset();la=np.zeros(12,np.float32);mh=.06
    for s in range(steps):
        if s%25==0:
            xm=np.asarray(mj_data.xmat[cat._base_body_id]).reshape(3,3)
            gv=-xm[2,:]
            av=mj_data.sensordata[cat._gyro_adr:cat._gyro_adr+3]
            jp=np.array([mj_data.qpos[cat._joint_qpos_adr[P[i] if op else i]] for i in range(12)],np.float32)
            jv=np.array([mj_data.qvel[cat._joint_qvel_adr[P[i] if op else i]] for i in range(12)],np.float32)
            obs=np.concatenate([av,gv,jp,jv,la]).astype(np.float32)
            act=cat.get_up_policy(obs[None,:])[0];la=np.asarray(act,np.float32)
            tg=np.zeros(12,np.float32)
            if ap:
                for k in range(12):tg[P[k]]=0.6*act[k]
            else:
                tg[:]=0.6*la
            for i in range(12):mj_data.ctrl[cat._actuator_ids[i]]=float(tg[i])
        mujoco.mj_step(mj_model,mj_data)
        h=float(mj_data.qpos[2])
        if h>mh:mh=h
    return mh
for op,ap in [(0,0),(0,1),(1,0),(1,1)]:
    print(f'obs_perm={op} act_perm={ap} max_h={test(op,ap):.4f}m',flush=True)
print("DONE")
