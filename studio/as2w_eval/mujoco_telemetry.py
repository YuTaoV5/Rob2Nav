"""Read simulation measurements without modifying the MuJoCo state."""
import mujoco as mj
import numpy as np


class PhysicalTelemetry:
    def __init__(self, model, base=None, joints=None):
        self.model = model
        free = next((i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE), None)
        self.base = int(model.jnt_bodyid[free]) if base is None and free is not None else base
        self.joints = list(joints) if joints is not None else [
            model.joint(i) for i in range(model.njnt)
            if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE
        ]
        self.q = [int(j.qposadr[0]) for j in self.joints]
        self.dq = [int(j.dofadr[0]) for j in self.joints]
        self.velocity = np.zeros(6)

    def sample(self, data):
        model = self.model
        imu = {}
        if self.base is not None:
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, self.base, self.velocity, 1)
            imu = dict(quaternion_wxyz=data.xquat[self.base].tolist(),
                       gyro_rad_s=self.velocity[:3].tolist(),
                       linear_velocity_m_s=self.velocity[3:].tolist(),
                       projected_gravity=(-data.xmat[self.base].reshape(3, 3)[2]).tolist())
        sensors = []
        for i in range(model.nsensor):
            kind = model.sensor_type[i]
            if kind not in (mj.mjtSensor.mjSENS_ACCELEROMETER, mj.mjtSensor.mjSENS_GYRO, mj.mjtSensor.mjSENS_FRAMEQUAT):
                continue
            adr, size = model.sensor_adr[i], model.sensor_dim[i]
            units = {mj.mjtSensor.mjSENS_ACCELEROMETER: "m/s²", mj.mjtSensor.mjSENS_GYRO: "rad/s"}
            sensors.append(dict(name=model.sensor(i).name, unit=units.get(kind, "wxyz"),
                                value=data.sensordata[adr:adr + size].tolist()))
        return dict(imu=imu, sensors=sensors, joints=[
            dict(name=j.name, position_rad=float(data.qpos[q]), velocity_rad_s=float(data.qvel[d]),
                 actuator_effort_nm=float(data.qfrc_actuator[d]))
            for j,q,d in zip(self.joints,self.q,self.dq)
        ])
