"""SDK2 codec and explicitly gated transport. No DDS initialization on import."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from .core import EFFORT
from .types import Snapshot,SDK_COMMIT,vector
from .calibration import Calibration


class LowCmdCodec:
    """Uses the real pinned SDK IDL factory and CRC, with no ChannelFactory call."""
    def __init__(self,calibration=None):
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        self.factory=unitree_go_msg_dds__LowCmd_;self.crc=CRC()
        self.calibration=(calibration or Calibration()).checked()
    def encode(self,command):
        tau=vector(command.tau)
        if np.any(abs(tau)>EFFORT+1e-8):raise ValueError('Unbounded motor torque')
        tau=self.calibration.torque_to_sdk(tau)
        msg=self.factory();msg.head[0]=0xFE;msg.head[1]=0xEF;msg.level_flag=0xFF;msg.gpio=0
        for i,motor in enumerate(msg.motor_cmd):
            motor.mode=0x01;motor.q=2.146e9;motor.dq=16000.;motor.kp=0.;motor.kd=0.;motor.tau=float(tau[i]) if i<12 else 0.
        msg.crc=self.crc.Crc(msg)
        return msg
    def damping(self,kd):
        if not np.isfinite(kd) or not 0<kd<=8:raise ValueError('Invalid qualified fallback damping')
        msg=self.factory();msg.head[0]=0xFE;msg.head[1]=0xEF;msg.level_flag=0xFF
        for i,m in enumerate(msg.motor_cmd):
            m.mode=1;m.q=2.146e9;m.dq=0.;m.kp=0.;m.kd=float(kd) if i<12 else 0.;m.tau=0.
        msg.crc=self.crc.Crc(msg);return msg


class Go2SdkTransport:
    """Read-only by default. All bundled skill profiles remain unqualified for hardware.

    A manual qualification record is an operational gate, not a cryptographic proof.
    This transport is never instantiated by simulation or offline codec tests.
    """
    def __init__(self,sdk_root,calibration=None):
        self.sdk_root=Path(sdk_root).resolve();self.lock=threading.Lock();self.latest=None
        self.last_tick=None;self.publisher=None;self.lease=None;self.permit=None;self.connected=False
        self.calibration=(calibration or Calibration()).checked();self.health=None
    def connect_readonly(self,interface):
        if self.connected:raise ValueError('DDS already initialized')
        pin=subprocess.check_output(['git','-C',str(self.sdk_root),'rev-parse','HEAD'],text=True).strip()
        if pin!=SDK_COMMIT:raise ValueError('Unreviewed SDK revision')
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        ChannelFactoryInitialize(0,interface)
        self.subscriber=ChannelSubscriber('rt/lowstate',LowState_);self.subscriber.Init(self._receive,10)
        self.connected=True
    def _receive(self,msg):
        tick=int(msg.tick)&0xffffffff
        with self.lock:
            if self.last_tick is not None and not 0<((tick-self.last_tick)&0xffffffff)<2**31:return
            self.last_tick=tick
            from .policies import rotation
            quat=vector(msg.imu_state.quaternion,4);rot=rotation(quat)
            self.serial='-'.join(map(str,msg.sn))
            self.health=dict(voltage=float(msg.power_v),temperature=[x.temperature for x in msg.motor_state[:12]],lost=[int(x.lost) for x in msg.motor_state[:12]])
            self.latest=Snapshot(time.monotonic(),tick,self.calibration.position_from_sdk([x.q for x in msg.motor_state[:12]]),
                self.calibration.vector_from_sdk([x.dq for x in msg.motor_state[:12]]),self.calibration.vector_from_sdk([x.tau_est for x in msg.motor_state[:12]]),
                vector(msg.imu_state.gyroscope,3),rot.T@np.array([0.,0.,-1.]),vector(msg.foot_force,4),quaternion=quat)
    def read(self,ignore_health=False):
        with self.lock:
            if self.latest is None:raise ValueError('No LowState received')
            if self.permit and not ignore_health:
                limits=self.permit['limits'];h=self.health
                if not np.isfinite([h['voltage'],*h['temperature']]).all() or not limits['voltage_min']<=h['voltage']<=limits['voltage_max'] or max(h['temperature'])>limits['temperature_max']:raise ValueError('Battery/temperature health gate')
                if any(((x-y)&0xffffffff)>limits['motor_lost_increment_max'] for x,y in zip(h['lost'],self.lost_baseline)):raise ValueError('Motor packet-loss health gate')
            return self.latest.checked(time.monotonic())
    def claim_writer(self,record,skill,lock_path='/tmp/rob2nav-go2-lowcmd.lock'):
        if not self.connected:raise ValueError('Read-only state acquisition must run first')
        if not skill.hardware_qualified:raise ValueError('Policy is not hardware qualified: '+skill.blocker)
        self.read()
        required=['robot_serial','firmware','sdk_commit','policy_sha256','calibration_sha256','watchdog_verified','reviewer','expires_unix','limits','recovery_kd']
        if any(k not in record for k in required):raise ValueError('Incomplete hardware qualification record')
        if record['robot_serial']!=self.serial or record['sdk_commit']!=SDK_COMMIT or record['policy_sha256']!=skill.sha256:raise ValueError('Device/SDK/policy qualification differs')
        if not record['firmware'] or not record['reviewer'] or len(record['calibration_sha256'])!=64 or record['watchdog_verified'] is not True or record['expires_unix']<=time.time():raise ValueError('Expired/incomplete physical qualification')
        if not self.calibration.verified or record['calibration_sha256']!=self.calibration.digest():raise ValueError('Device calibration is unverified/different')
        limits=record['limits']
        if not 0<limits['voltage_min']<limits['voltage_max'] or not 0<limits['temperature_max']<100 or not 0<=limits['motor_lost_increment_max']<=10 or not 0<record['recovery_kd']<=8:raise ValueError('Invalid device health/recovery limits')
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        m=MotionSwitcherClient();m.SetTimeout(2.);m.Init();status,mode=m.CheckMode()
        if status!=0 or mode.get('name'):raise ValueError('Another motion service owns the robot; no automatic StandDown/ReleaseMode is performed')
        import fcntl
        fd=os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except Exception:os.close(fd);raise
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            self.publisher=ChannelPublisher('rt/lowcmd',LowCmd_);self.publisher.Init();self.codec=LowCmdCodec(self.calibration)
            self.lease=fd;self.permit=dict(record)
            self.lost_baseline=self.health['lost'].copy()
        except Exception:os.close(fd);raise
    def write(self,command):
        if self.publisher is None:raise ValueError('Read-only transport; writer not qualified/claimed')
        now=time.monotonic();self.read(ignore_health=command.phase=='FAULT')
        if not 0<=now-command.stamp<=.01:raise ValueError('Expired motor command')
        if self.permit['expires_unix']<=time.time():raise ValueError('Hardware permit expired')
        before=time.monotonic();self.publisher.Write(self.codec.encode(command))
        if time.monotonic()-before>.01:raise TimeoutError('SDK publisher exceeded control deadline; independent device watchdog is required')
    def emergency_damping(self):
        if self.publisher is None or self.permit is None:raise ValueError('No qualified writer exists')
        self.publisher.Write(self.codec.damping(self.permit['recovery_kd']))
    def close(self):
        # Closing a publisher is not a substitute for a device-qualified recovery.
        if self.publisher is not None and hasattr(self.publisher,'Close'):self.publisher.Close()
        if self.connected and hasattr(self.subscriber,'Close'):self.subscriber.Close()
        if self.lease is not None:os.close(self.lease);self.lease=None
        self.publisher=None
