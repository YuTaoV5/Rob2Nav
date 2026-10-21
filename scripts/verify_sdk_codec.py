"""Real SDK IDL/CRC verification only. DDS factory/publishers are forbidden here."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rob2nav.deployment.types import MotorCommand
from rob2nav.deployment.core import NEUTRAL
from rob2nav.deployment.unitree import LowCmdCodec,Go2SdkTransport


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--sdk-root',type=Path,required=True);a=p.parse_args()
    with patch('unitree_sdk2py.core.channel.ChannelFactoryInitialize',side_effect=AssertionError('No DDS allowed')),patch('unitree_sdk2py.core.channel.ChannelPublisher',side_effect=AssertionError('No publisher allowed')):
        codec=LowCmdCodec();tau=np.arange(12)*.1
        c=MotorCommand(tau,NEUTRAL,np.zeros(12),np.ones(12)*40,np.ones(12),'ACTIVE',7)
        msg=codec.encode(c)
        assert [m.tau for m in msg.motor_cmd[:12]]==tau.tolist()
        assert all(m.kp==0 and m.kd==0 for m in msg.motor_cmd), 'PD would be applied twice'
        assert all(m.tau==0 for m in msg.motor_cmd[12:])
        assert msg.crc==codec.crc.Crc(msg)
        crc=int(msg.crc);msg.motor_cmd[0].tau+=.1;assert codec.crc.Crc(msg)!=crc
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
        transport=Go2SdkTransport(a.sdk_root);state=unitree_go_msg_dds__LowState_();state.tick=1
        state.imu_state.quaternion[:]=[1.,0.,0.,0.];state.foot_force[:]=[35,35,35,35]
        for i in range(12):state.motor_state[i].q=float(NEUTRAL[i])
        transport._receive(state);first=transport.latest.stamp
        transport._receive(state);assert transport.latest.stamp==first,'duplicate tick refreshed stale state'
        assert transport.read().stable()
        durations=[]
        for _ in range(1000):
            start=time.perf_counter();codec.encode(c);durations.append(time.perf_counter()-start)
        report=dict(real_sdk_idl=True,real_sdk_crc=True,wire_motors=len(msg.motor_cmd),serialized_bytes=len(msg.serialize()),
            double_pd=False,crc_detected_mutation=True,duplicate_lowstate_rejected=True,
            codec_milliseconds_p99=float(np.percentile(durations,99)*1000),codec_milliseconds_max=max(durations)*1000,
            channel_factory_calls=0,publisher_calls=0,hardware_connected=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
