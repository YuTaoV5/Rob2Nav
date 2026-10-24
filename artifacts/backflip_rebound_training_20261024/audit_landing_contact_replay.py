"""Frozen-model read-only contact/torque replay; run with normal simulate arguments."""
import hashlib
import json
from pathlib import Path
from skills.backflip.simulate_landing import LandingEvaluation
from skills.backflip.simulate import main

class ContactReplay(LandingEvaluation):
    def configure(self,*args):
        super().configure(*args)
        self.contact_rows=[]
        self.contact_first={}
    def sample(self,model,data,frame,qids,dids):
        super().sample(model,data,frame,qids,dids)
        for c in frame.unexpected:
            self.contact_first.setdefault(str(c['geom']),dict(time=float(data.time),**c))
        if 1.9<=data.time<=2.35:
            self.contact_rows.append(dict(time=float(data.time),phase=int(self.runtime.state.phase[0]),
                q=data.qpos[qids].tolist(),dq=data.qvel[dids].tolist(),
                root_position=data.qpos[:3].tolist(),omega=data.qvel[3:6].tolist(),
                actuator_torque=data.qfrc_actuator[dids].tolist(),constraint_torque=data.qfrc_constraint[dids].tolist(),
                bias_torque=data.qfrc_bias[dids].tolist(),foot_forces=frame.forces,unexpected=frame.unexpected,
                reconstructed_servo_target=self.last_target.tolist()))
    def report(self):
        result=super().report()
        report=dict(scope='read-only post-step contact and generalized torque sampling; no state writes or controller change',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    joint_names=list(self.module.JOINT_NAMES),first_unexpected=list(self.contact_first.values()),rows=self.contact_rows)
        (self.args.output/'contact_window.json').write_text(json.dumps(report,indent=2))
        return result

if __name__=='__main__': main(extension=ContactReplay())
