"""Procedural task geometry and static-map routing. No robot state writes."""
import heapq
import math
import xml.etree.ElementTree as ET
import numpy as np

TASK_SCENES=('rooms_dynamic','follow_obstacles','follow_stairs','follow_sand','follow_mixed')
COLORS=((.92,.55,.12,1),(.17,.57,.85,1),(.66,.33,.78,1))

def populate(world,scene,cfg,obstacle_names):
    meta=dict(goal=([9.,2.0] if scene=='rooms_dynamic' else [8.,0.]),surfaces=[],terrain='flat',people=[])
    def box(name,x,y,sx,sy,height=.65,obstacle=True,rgba='.65 .48 .35 1',friction='.8 .02 .01'):
        if obstacle:obstacle_names.append(name)
        ET.SubElement(world,'geom',name=name,type='box',pos=f'{x} {y} {height/2}',size=f'{sx} {sy} {height/2}',rgba=rgba,friction=friction,group='0',priority='1')
    if scene=='rooms_dynamic':
        for x in (3.,6.):
            for lo,hi in [(-3.,-2.15),(-.65,.65),(2.15,3.)]:box(f'partition_{x}_{lo}',x,(lo+hi)/2,.08,(hi-lo)/2,1.25)
        for lo,hi in [(-2.,-1.25),(1.25,3.75),(5.25,7.25),(8.75,10.)]:box(f'room_divider_{lo}',(lo+hi)/2,0,(hi-lo)/2,.08,1.25)
        paths=[[[1.7,-1.4],[4.5,-1.4],[4.5,1.4],[8.,1.4],[4.5,1.4],[4.5,-1.4]],
               [[1.4,1.5],[1.4,2.4],[-.8,2.4],[-.8,1.5]],
               [[8.,1.8],[8.,-1.8],[7.,-1.8],[8.,-1.8]]]
    else:
        for i,(x,y) in enumerate([(2.,1.5),(5.8,-1.5),(7.5,1.5)]):box(f'furniture_{i}',x,y,.3,.4,.65)
        # The leader continues beyond the terrain exit before returning along
        # the outer aisle, so a following robot can clear the entire surface.
        paths=[[[1.7,0],[9.4,0],[9.4,2.4],[1.,2.4],[1.,0]],[[4.,1.6],[4.,-1.6]],[[7.6,-1.4],[7.6,1.4]]]
        if scene in ('follow_stairs','follow_mixed'):
            surfaces=[]
            for i in range(4):surfaces.append((3.+i*.35,3.+(i+1)*.35,(i+1)*.025))
            surfaces.append((4.4,5.,.10))
            for i in range(4):surfaces.append((5.+i*.35,5.+(i+1)*.35,(3-i)*.025))
            for i,(lo,hi,h) in enumerate(surfaces):
                if h>0:box(f'stair_{i}',(lo+hi)/2,0,(hi-lo)/2,2.88,h,False,rgba='.55 .62 .68 1')
            meta['surfaces']=surfaces;meta['terrain']='rigid stairs: 4 x 0.025m, tread0.35m; not standard staircase qualification'
        if scene in ('follow_sand','follow_mixed'):
            lo,hi=(2.7,6.4) if scene=='follow_sand' else (6.6,7.6)
            # Rigid contact roughness and low friction only; no granular soil/DEM.
            for i,x in enumerate(np.arange(lo,hi,.25)):
                for j,y in enumerate(np.arange(-2.75,2.76,.25)):
                    h=.008+.004*((i+2*j)%3)
                    box(f'sand_approx_{i}_{j}',x,y,.125,.125,h,False,rgba='.75 .63 .38 1',friction='.38 .015 .003')
            meta['terrain']+='; sand approximation: rough rigid tiles, friction0.38, no particles/sinkage'
            meta['sand_region']=[lo,hi]
    difficulty=cfg.get('task_difficulty',2)
    for i,path in enumerate(paths):
        ident=f'person_{i+1:02d}';body=ET.SubElement(world,'body',name=ident,mocap='true',pos=f'{path[0][0]} {path[0][1]} 0')
        rgba=' '.join(map(str,COLORS[i]))
        for suffix,typ,attrs in [('torso','capsule',dict(fromto='0 0 .65 0 0 1.18',size='.19')),
                                  ('head','sphere',dict(pos='0 0 1.43',size='.14')),
                                  ('leg_l','capsule',dict(fromto='0 .10 .08 0 .10 .65',size='.07')),
                                  ('leg_r','capsule',dict(fromto='0 -.10 .08 0 -.10 .65',size='.07'))]:
            name=ident+'_'+suffix;obstacle_names.append(name)
            ET.SubElement(body,'geom',name=name,type=typ,rgba=rgba,group='0',friction='.8 .02 .01',**attrs)
        meta['people'].append(dict(id=ident,path=path,speed=(.12+.03*difficulty) if i==0 else (.20+.08*difficulty),color=COLORS[i]))
    return meta

class StaticRoute:
    """Known building floor plan only. Moving person truth is never an A* obstacle."""
    def __init__(self,boxes):self.boxes=boxes;self.res=.2
    def free(self,p):
        return -1.45<=p[0]<=9.4 and -2.5<=p[1]<=2.5 and all(abs(p[0]-b['x'])>=b['sx']+.37 or abs(p[1]-b['y'])>=b['sy']+.37 for b in self.boxes)
    def line(self,a,b):
        n=max(2,int(math.dist(a,b)/.05)+1)
        return all(self.free(np.asarray(a)+(np.asarray(b)-a)*t) for t in np.linspace(0,1,n))
    def carrot(self,start,path):
        desired=np.asarray(path[1]);best=0.
        for a,b in zip(path,path[1:]):
            for t in np.linspace(0,1,max(2,int(math.dist(a,b)/.15))):
                p=np.asarray(a)+(np.asarray(b)-a)*t;d=math.dist(start,p)
                if best<d<=3. and self.line(start,p):desired=p;best=d
        return desired
    def path(self,start,goal):
        if not self.free(goal):return []
        if self.line(start,goal):return [list(start),list(goal)]
        cell=lambda p:(round(p[0]/self.res),round(p[1]/self.res))
        xy=lambda p:np.array(p)*self.res
        s,g=cell(start),cell(goal);todo=[(0,s)];cost={s:0};prev={};done=set()
        while todo and len(done)<5000:
            _,u=heapq.heappop(todo)
            if u in done:continue
            done.add(u)
            if u==g:
                path=[list(goal)];v=u
                while v!=s:path.append(xy(v).tolist());v=prev[v]
                path.append(list(start));path.reverse();smooth=[path[0]];i=0
                while i<len(path)-1:
                    j=len(path)-1
                    while j>i+1 and not self.line(path[i],path[j]):j-=1
                    smooth.append(path[j]);i=j
                return smooth
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                v=(u[0]+dx,u[1]+dy)
                if not self.line(xy(u),xy(v)):continue
                nc=cost[u]+math.hypot(dx,dy)
                if nc<cost.get(v,1e9):cost[v]=nc;prev[v]=u;heapq.heappush(todo,(nc+math.dist(v,g),v))
        return []

def follow_goal(robot,person,velocity,distance):
    delta=np.asarray(person)-np.asarray(robot);norm=np.linalg.norm(delta)
    direction=np.asarray(velocity)
    if np.linalg.norm(direction)<.05:direction=delta
    direction=direction/max(np.linalg.norm(direction),1e-6)
    return np.asarray(person)-direction*distance,norm

def pedestrian_guard(command,rotation,robot,observations):
    """Reduce velocity until the visible-track 2s constant-velocity check clears."""
    cmd=np.asarray(command,dtype=float).copy();visible=[p for p in observations if p['visible']]
    def safe(scale):
        world=rotation[:2,:2]@(cmd[:2]*scale)
        for person in visible:
            delta=np.asarray(person['position'])-np.asarray(robot);rv=np.asarray(person['velocity'])-world
            t=np.clip(-np.dot(delta,rv)/max(np.dot(rv,rv),1e-8),0,2.)
            if np.linalg.norm(delta)<.85 or np.linalg.norm(delta+rv*t)<.80:return False
        return True
    for factor in (1.,.75,.5,.25,0.):
        if safe(factor):
            cmd[:2]*=factor
            return cmd,'' if factor==1 else 'predicted_person_slowdown'
    cmd[:2]=0;cmd[2]*=.3;return cmd,'predicted_person_conflict'
