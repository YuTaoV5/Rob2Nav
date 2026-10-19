"""Bounded geometry/command probes, not a port of SCAN or a trained SEA policy."""
import heapq,math,time

SCENES={
 'clutter':[(3,4,0,3,0,1),(3,4,4.2,8,0,1),(6,7,0,5,0,1),(8.5,9.2,3,8,0,1)],
 'narrow':[(2,9,0,3.4,0,1),(2,9,4.6,8,0,1)],
 'overhang':[(3,5,2,6,.85,1.5),(6,7,0,5,0,1)],
 'dead_end':[(4,5,1,6,0,1),(1.5,5,1,2,0,1),(1.5,5,5,6,0,1)]}


def footprint(x,y,yaw,length=.6,width=.3,margin=.05):
    radius=width/2+margin;offset=max(0,(length-width)/2)
    return [(x+s*offset*math.cos(yaw),y+s*offset*math.sin(yaw),radius) for s in (-1,1)]


def collides(x,y,yaw,boxes,length,width,margin,body_z=.32,body_height=.24):
    for cx,cy,r in footprint(x,y,yaw,length,width,margin):
        if cx-r<0 or cx+r>12 or cy-r<0 or cy+r>8:return True
        for x0,x1,y0,y1,z0,z1 in boxes:
            if body_z+body_height/2<z0 or body_z-body_height/2>z1:continue
            dx=cx-max(x0,min(x1,cx));dy=cy-max(y0,min(y1,cy))
            if dx*dx+dy*dy<=r*r:return True
    return False


def shield(v,position,obstacles,radius=.3,alpha=2.):
    """Projection onto instantaneous circle-distance CBF halfspaces, SI units.

    Reference velocity integrator only. No quadruped dynamics or safety guarantee.
    An already intersecting or infeasible state returns stop with a reason.
    """
    out=list(v);constraints=[]
    for x,y,r in obstacles:
        dx,dy=position[0]-x,position[1]-y;h=dx*dx+dy*dy-(r+radius)**2
        if h<=0:return dict(velocity=[0.,0.],feasible=False,reason='inside_margin')
        constraints.append((2*dx,2*dy,-alpha*h))
    for _ in range(40):
        for a,b,c in constraints:
            gap=c-a*out[0]-b*out[1]
            if gap>0:
                norm=a*a+b*b
                if norm<1e-12:return dict(velocity=[0.,0.],feasible=False,reason='degenerate')
                out[0]+=gap*a/norm;out[1]+=gap*b/norm
    feasible=all(a*out[0]+b*out[1]>=c-1e-7 for a,b,c in constraints)
    return dict(velocity=out if feasible else [0.,0.],feasible=feasible,reason='projected' if feasible else 'infeasible')


def plan(scene='clutter',length=.6,width=.3,margin=.05,body_z=.32,start=(1.,4.),goal=(11.,4.)):
    if scene not in SCENES:raise ValueError('Unknown scene')
    if not (.3<=length<=1.2 and .15<=width<=.6 and 0<=margin<=.3 and .2<=body_z<=.8):raise ValueError('Invalid robot geometry')
    for p in (start,goal):
        if len(p)!=2 or not all(math.isfinite(a) for a in p):raise ValueError('Invalid waypoint')
    boxes=SCENES[scene];resolution=.2;directions=[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    origin=(round(start[0]/resolution),round(start[1]/resolution),0);target=(round(goal[0]/resolution),round(goal[1]/resolution))
    def valid(state):return not collides(state[0]*resolution,state[1]*resolution,state[2]*math.pi/4,boxes,length,width,margin,body_z)
    begin=time.perf_counter();queue=[(0,0,origin)];cost={origin:0};parent={};end=None;expanded=0
    if not valid(origin):queue=[]
    while queue and expanded<25000:
        _,g,s=heapq.heappop(queue)
        if g!=cost.get(s):continue
        expanded+=1
        if s[:2]==target:end=s;break
        # Rotation in place plus heading-constrained forward/backward translations.
        moves=[(s[0],s[1],(s[2]+d)%8,.15) for d in (-1,1)]
        dx,dy=directions[s[2]]
        moves += [(s[0]+sign*dx,s[1]+sign*dy,s[2],resolution*math.hypot(dx,dy)*(1 if sign==1 else 1.3)) for sign in (1,-1)]
        for nx,ny,nh,c in moves:
            n=(nx,ny,nh)
            if not (0<=nx<=60 and 0<=ny<=40) or not valid(n):continue
            delta=((nh-s[2]+4)%8-4)*math.pi/4
            if any(collides((s[0]+(nx-s[0])*t)*resolution,(s[1]+(ny-s[1])*t)*resolution,s[2]*math.pi/4+delta*t,boxes,length,width,margin,body_z) for t in (.25,.5,.75)):continue
            ng=g+c
            if ng>=cost.get(n,float('inf')):continue
            cost[n]=ng;parent[n]=s;heapq.heappush(queue,(ng+resolution*math.hypot(nx-target[0],ny-target[1]),ng,n))
    path=[]
    while end is not None:path.append([end[0]*resolution,end[1]*resolution,body_z,end[2]*math.pi/4]);end=parent.get(end)
    path.reverse()
    return dict(scene=scene,boxes=boxes,path=path,start=start,goal=goal,length=length,width=width,margin=margin,body_z=body_z,
        success=bool(path),expanded=expanded,planning_ms=1000*(time.perf_counter()-begin),
        path_length_m=sum(math.dist(a[:2],b[:2]) for a,b in zip(path,path[1:])),
        provenance='Rob2Nav reference SE(2) A* with height-filtered boxes and twin-circle footprint. Not SCAN-Planner native optimization, not SEA-Nav policy, not leg dynamics.',
        limitations=['Static exact geometry supplied to probe, not onboard perception','No sliding voxel map or global-route provider','No ground following, jumping or stair controller','No learned adaptive CBF or SEA training reproduced'])


def integrations():
    return [dict(id='scan',name='SCAN-Planner',status='source audited / native ROS runtime not connected',
        url='https://github.com/wuyi2121/SCAN-Planner',layer='3D occupancy / spatial local trajectory',
        commit='348e8a590a50a5a6bbab8d8c6dcfd171f009be26',runtime='Ubuntu20.04 / ROS Noetic; native build pending',
        inputs='localized body pose, sensor pose, depth or LiDAR, goal/reference route',outputs='native trajectory/controller interface requires source adapter',
        distinction='Online local mapping is allowed; long-range guidance is a separate input, not map-free perception'),
        dict(id='sea',name='SEA-Nav',status='source audit / native policy not connected',url='https://11chens.github.io/sea-nav/',
        code='https://github.com/11chens/SEA-Nav-Code',commit='560de73c094a750a6a358bc1920ebc9ef1d9f3c5',runtime='Isaac Gym legacy / ROS2; navigation checkpoint absent in audited tree',
        layer='learned navigation velocity policy and adaptive CBF',inputs='10 x 55 history: 12 proprio + 41 log2 rays (240 degree sector) + 2 goal; deployment RPLIDAR/BreezySLAM',
        outputs='velocity commands, not joint-level obstacle locomotion',distinction='Minute-level training is the authors’ reported setup, not a promise for this GPU/robot/task')]
