"""Map body-relative Go2 foot trajectories to AS2-W wheel-center trajectories."""
import numpy as np

GO2_NEUTRAL = np.array(((.1,.8,-1.5),(-.1,.8,-1.5),(.1,1.,-1.5),(-.1,1.,-1.5)))
SIDE = np.array((1.,-1.,1.,-1.))


def endpoint(q, hip_offset, link_length):
    hip, thigh, calf = np.moveaxis(q,-1,0)
    planar = -link_length*(np.cos(thigh)+np.cos(thigh+calf))
    x = -link_length*(np.sin(thigh)+np.sin(thigh+calf))
    lateral = SIDE*hip_offset
    y = np.cos(hip)*lateral-np.sin(hip)*planar
    z = np.sin(hip)*lateral+np.cos(hip)*planar
    return np.stack((x,y,z),axis=-1)


def inverse_endpoint(points, hip_offset, link_length):
    x, y, z = points.T
    lateral = SIDE*hip_offset
    planar = -np.sqrt(np.maximum(y*y+z*z-lateral*lateral,1.e-8))
    hip = np.arctan2(z,y)-np.arctan2(planar,lateral)
    hip = (hip+np.pi)%(2.*np.pi)-np.pi
    cosine = (x*x+planar*planar)/(2.*link_length**2)-1.
    calf = -np.arccos(np.clip(cosine,-.999,.999))
    thigh = np.arctan2(-x,-planar)-np.arctan2(np.sin(calf),1.+np.cos(calf))
    return np.column_stack((hip,thigh,calf))


class Go2WheelRetarget:
    def __init__(self, as2w_neutral):
        self.go2_neutral_xyz = endpoint(GO2_NEUTRAL,.0955,.213)
        self.as2w_neutral_xyz = endpoint(as2w_neutral,.1054,.212)

    def observation(self, as2w_q):
        relative = endpoint(as2w_q,.1054,.212)-self.as2w_neutral_xyz
        return inverse_endpoint(self.go2_neutral_xyz+relative,.0955,.213)-GO2_NEUTRAL

    def target(self, go2_action):
        relative = endpoint(GO2_NEUTRAL+.25*go2_action,.0955,.213)-self.go2_neutral_xyz
        # Relative vertical motion is the same at the wheel center and tread.
        return inverse_endpoint(self.as2w_neutral_xyz+relative,.1054,.212)


def project_wheel_targets(initial, targets, bounds):
    """Fit wheel centers while keeping every joint inside its permitted range."""
    lo, hi = bounds[...,0], bounds[...,1]
    q = np.clip(initial,lo,hi)
    eps = 1.e-5
    axes = np.eye(3)*eps
    for _ in range(12):
        current = endpoint(q,.1054,.212)
        jacobian = np.stack([
            (endpoint(q+axis,.1054,.212)-current)/eps for axis in axes
        ],axis=-1)
        residual = targets-current
        # Preserve vertical clearance when a lateral or longitudinal target is infeasible.
        weight = np.array((1.,1.,2.))
        gradient = np.einsum("...ij,...i->...j",jacobian,residual*weight)
        hessian = np.einsum("...ij,...ik,i->...jk",jacobian,jacobian,weight)
        delta = np.linalg.solve(hessian+.003*np.eye(3),gradient[...,None])[...,0]
        q = np.clip(q+np.clip(delta,-.10,.10),lo,hi)
    return q
