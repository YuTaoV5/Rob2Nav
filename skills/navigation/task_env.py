"""Free-base robot dynamics plus scripted collidable pedestrians."""
import math
import mujoco
import numpy as np
from go2_nav.env import Go2MaplessEnv

class TaskEnvironment(Go2MaplessEnv):
    def _world(self):
        super()._world();self.people=[];self.actors_active=False
        for item in getattr(self,'terrain_meta',{}).get('people',[]):
            body=self.model.body(item['id']);p=dict(item,body=int(body.id),mocap=int(body.mocapid[0]),index=1,
                position=np.array(item['path'][0],dtype=float),velocity=np.zeros(2))
            self.people.append(p)
        self.static_boxes=[(self.model.geom_pos[g,:2].copy(),self.model.geom_size[g,:2].copy()) for g in self.obstacle_ids if self.model.geom_bodyid[g]==0]
        self.robot_geoms=set()
        for g in range(self.model.ngeom):
            b=int(self.model.geom_bodyid[g])
            while b:
                if b==self.base_id:self.robot_geoms.add(g);break
                b=int(self.model.body_parentid[b])
    def actor_free(self,p):
        return -1.7<p[0]<9.65 and -2.7<p[1]<2.7 and all(np.any(np.abs(p-center)>size+.23) for center,size in self.static_boxes)
    def ground(self,x):
        return max([h for lo,hi,h in getattr(self,'terrain_meta',{}).get('surfaces',[]) if lo<=x<hi]+[0.])
    def _control(self,command):
        if getattr(self,'actors_active',False):
            for p in self.people:
                path=p['path'];delta=np.asarray(path[p['index']])-p['position'];n=np.linalg.norm(delta)
                if n<.08:p['index']=(p['index']+1)%len(path);delta=np.asarray(path[p['index']])-p['position'];n=np.linalg.norm(delta)
                velocity=delta/max(n,1e-9)*p['speed']
                p['waiting']=False
                if (getattr(self,'follow_mode',False) and p['id']==getattr(self,'selected_person','person_01')
                    and self.cfg.get('task_difficulty',2)<3
                    and np.linalg.norm(p['position']-self.data.qpos[:2])>getattr(self,'guide_gap',2.5)):
                    velocity*=0;p['waiting']=True
                proposed=p['position']+velocity*self.cfg['control_dt']
                # Cooperative pedestrian sidestep prevents two stop-only agents
                # deadlocking. It is scripted scenario behaviour, not robot planning.
                away=p['position']-self.data.qpos[:2];gap=np.linalg.norm(away)
                if np.linalg.norm(proposed-self.data.qpos[:2])<.95 and gap>.01:
                    away/=gap;tangent=np.array([-away[1],away[0]])
                    velocity*=0
                    for side in (1,-1,0):
                        candidate=(away*.12+side*tangent*.22)
                        point=p['position']+candidate*self.cfg['control_dt']
                        if self.actor_free(point):velocity=candidate;break
                if not self.actor_free(p['position']+velocity*self.cfg['control_dt']):velocity*=0
                p['position']+=velocity*self.cfg['control_dt'];p['velocity']=velocity
                self.data.mocap_pos[p['mocap']]=[*p['position'],self.ground(p['position'][0])]
                yaw=math.atan2(velocity[1],velocity[0]) if np.linalg.norm(velocity)>.01 else 0
                self.data.mocap_quat[p['mocap']]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        super()._control(command)
    def _collision(self):
        for c in self.data.contact:
            a,b=int(c.geom1),int(c.geom2)
            if (a in self.obstacle_ids and b in self.robot_geoms) or (b in self.obstacle_ids and a in self.robot_geoms):return True
        return False
    def observations(self):
        origin=self.data.xpos[self.base_id]+self.rotation@np.asarray(self.cfg['lidar_position']);tracks=[]
        for p in self.people:
            point=np.array([*p['position'],self.ground(p['position'][0])+1.0]);delta=point-origin;n=np.linalg.norm(delta);hit=np.zeros(1,dtype=np.int32)
            visible=False
            if .2<n<self.cfg['max_range'] and -.12<math.asin(delta[2]/n)<.86:
                mujoco.mj_ray(self.model,self.data,origin,delta/n,self.ray_groups,1,self.base_id,hit)
                visible=hit[0]>=0 and int(self.model.geom_bodyid[hit[0]])==p['body']
            tracks.append(dict(id=p['id'],position=p['position'].tolist(),velocity=p['velocity'].tolist(),visible=visible,color=p['color'],guide_waiting=p.get('waiting',False)))
        return tracks
