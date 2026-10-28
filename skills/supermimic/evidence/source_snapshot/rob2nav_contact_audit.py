"""Read-only contact classification for MuJoCo Warp physical substeps."""
import torch
import warp as wp
import mujoco
def forbidden_mask(geom,world,dist,nacon,robot_nonfoot,ground,num_envs):
 ids=torch.arange(geom.shape[0],device=geom.device)
 valid=(ids<nacon)&(world>=0)&(world<num_envs)&(geom[:,0]>=0)&(geom[:,1]>=0)&(dist<=0)
 a=geom[:,0].clamp(min=0,max=len(ground)-1).long()
 b=geom[:,1].clamp(min=0,max=len(ground)-1).long()
 return valid&((robot_nonfoot[a]&ground[b])|(robot_nonfoot[b]&ground[a]))
class BodyContactAudit:
 def __init__(self,env):
  self.env=env;m=env.sim.mj_model;dev=env.device
  self.names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or str(i) for i in range(m.ngeom)]
  robot_ids=set(env.scene["robot"].indexing.geom_ids.tolist())
  self.robot_nonfoot=torch.tensor([i in robot_ids and not self.names[i].endswith("_foot_collision") for i in range(m.ngeom)],device=dev)
  self.ground=torch.tensor([i not in robot_ids and (m.geom_type[i] in (mujoco.mjtGeom.mjGEOM_PLANE,mujoco.mjtGeom.mjGEOM_HFIELD) or "terrain" in self.names[i]) for i in range(m.ngeom)],device=dev)
  if not bool(self.ground.any()):raise RuntimeError("No terrain geometries classified")
  self.count=torch.zeros(env.num_envs,device=dev,dtype=torch.long);self.samples=0;self.first=None;self.overflow=False
  self.geom=wp.to_torch(env.sim.wp_data.contact.geom);self.world=wp.to_torch(env.sim.wp_data.contact.worldid)
  self.dist=wp.to_torch(env.sim.wp_data.contact.dist);self.nacon=wp.to_torch(env.sim.wp_data.nacon)
 def update(self):
  mask=forbidden_mask(self.geom,self.world,self.dist,self.nacon[0],self.robot_nonfoot,self.ground,self.env.num_envs)
  worlds=self.world[mask].long();self.count.scatter_add_(0,worlds,torch.ones_like(worlds))
  if self.first is None and bool(mask.any()):
   k=int(mask.nonzero()[0]);a,b=self.geom[k].tolist()
   self.first={"sample":self.samples,"environment":int(self.world[k]),"geoms":[self.names[a],self.names[b]],"distance":float(self.dist[k])}
  if bool(self.nacon[0]>len(self.geom)):self.overflow=True
  self.samples+=1
 def summary(self):
  return {"physical_samples":self.samples,"nonfoot_ground_contact_samples":int(self.count.sum()),
   "environments_with_nonfoot_ground_contact":int((self.count>0).sum()),"first_contact":self.first,
   "contact_buffer_overflow":self.overflow,"ground_geoms":[n for n,b in zip(self.names,self.ground.tolist()) if b],
   "criterion":"active raw collision contacts with dist<=0; ground versus any robot geom except named foot; no state writes"}

