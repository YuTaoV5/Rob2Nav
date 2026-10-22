"""World-fixed visual references, added only to mjvScene (never to physics)."""
import numpy as np
import mujoco


def reference_ground(scene,model,data):
    # A quarter-metre grid stays fixed in world coordinates as the camera follows.
    # MuJoCo visual geoms have no collision, inertia, sensor or controller effect.
    center=np.floor((np.asarray(data.qpos[:2]) if len(data.qpos)>=2 else np.zeros(2))/2.)*2.
    for geom in scene.geoms[:scene.ngeom]:
        if geom.type==mujoco.mjtGeom.mjGEOM_PLANE:
            geom.rgba[:]=[.64,.69,.73,1.]
    def geom(kind,size,pos,color):
        if scene.ngeom>=scene.maxgeom:return None
        g=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(g,kind,np.asarray(size),np.asarray(pos),np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
        g.category=mujoco.mjtCatBit.mjCAT_DECOR
        return g
    for i in range(-8,8):
        for j in range(-8,8):
            if (i+j)%2==0:
                geom(mujoco.mjtGeom.mjGEOM_BOX,[.25,.25,.0004],
                    [center[0]+(i+.5)*.5,center[1]+(j+.5)*.5,.0005],[.77,.80,.81,1.])
    for axis in (0,1):
        for k in range(-16,17):
            a=np.r_[center-4.,.002];b=np.r_[center+4.,.002]
            a[axis]=b[axis]=center[axis]+k*.25
            major=k%4==0;color=[.34,.42,.48,1.] if major else [.51,.58,.62,1.]
            g=geom(mujoco.mjtGeom.mjGEOM_LINE,[0,0,0],[0,0,0],color)
            if g is not None:mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,1.5 if major else .7,a,b)
    for a,b,color in [([-1.,0.,.005],[1.,0.,.005],[.80,.27,.21,1.]),([0.,-1.,.005],[0.,1.,.005],[.18,.42,.70,1.])]:
        g=geom(mujoco.mjtGeom.mjGEOM_LINE,[0,0,0],[0,0,0],color)
        if g is not None:mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,3.,np.array(a),np.array(b))
