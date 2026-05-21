import mujoco
COUCH_POS=(1.0,0.0,0.20)
COUCH_HALF=(0.40,0.60,0.20)
COUCH_RGBA=(0.45,0.30,0.20,1.0)
def add_couch_spec_fn(spec):
    body=spec.worldbody.add_body(name="couch",pos=list(COUCH_POS))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=list(COUCH_HALF),rgba=list(COUCH_RGBA))
