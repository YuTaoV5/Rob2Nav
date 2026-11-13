"""Validate AS2-W mode transitions and policy inference without motor output."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from as2w_deploy_preflight import validate_binding
from as2w_mode_switch import Feedback, ModeSwitch, SKILLS


class Policy:
    def __init__(self, directory, shape):
        path = directory/"policy.onnx"
        report = json.loads((directory/"onnx-parity.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report["onnx_sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        if report["max_action_error"]>1e-4:
            raise ValueError(f"ONNX parity exceeds tolerance: {path}")
        self.provenance = {"onnx":str(path.resolve()),"onnx_sha256":report["onnx_sha256"],
                           "checkpoint_sha256":report.get("checkpoint_sha256")}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.model = ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
        obs, output = self.model.get_inputs(),self.model.get_outputs()
        width, actions = shape
        if (len(obs)!=1 or len(output)!=1
                or (obs[0].name,obs[0].shape,obs[0].type)!=("obs",[1,width],"tensor(float)")
                or (output[0].name,output[0].shape,output[0].type)!=("actions",[1,actions],"tensor(float)")):
            raise ValueError(f"Unexpected policy contract: {path}")
        self.width = width

    def run(self, observation, *, name, centered, reset):
        frame = np.array(observation,dtype=np.float32,copy=True)
        if frame.shape!=(self.width,) or not np.isfinite(frame).all():
            raise ValueError("Invalid policy observation")
        if centered and name in ("rolling","stepping"):
            frame[:3] = (-.4,0.,0.) if name=="rolling" else 0.
            if name=="rolling":
                frame[64] = 1.
        if reset:
            if name=="rolling":
                frame[42:58] = 0.
                frame[58:60] = (0.,1.)
                frame[60:64] = 0.
            elif name=="stepping":
                frame[33:45] = 0.
                frame[45:47] = (0.,1.)
            else:
                frame[30:42] = 0.
        action, = self.model.run(["actions"],{"obs":frame[None]})
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite policy action")
        return action[0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference",type=Path)
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    for name in SKILLS:
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.input.resolve()==args.output.resolve():
        parser.error("Use a separate replay output")
    validate_binding(json.loads(args.reference.read_text()),False)
    shapes = {"rolling":(65,16),"stepping":(47,12),"getup":(42,12)}
    models = {name:Policy(getattr(args,name),shape) for name,shape in shapes.items()}
    supervisor = ModeSwitch()
    pending_reset = None
    with args.input.open() as source,args.output.open("x") as out:
        for line in source:
            frame = json.loads(line)
            decision = supervisor.step(Feedback(**frame["feedback"]),frame.get("request"))
            report = asdict(decision)
            if decision.reset_policy_history:
                pending_reset = decision.policy
            if decision.policy is not None and decision.weight>0.:
                name = decision.policy
                reset = pending_reset == name
                centered = not decision.joystick_enabled
                report["raw_actor_output"] = models[name].run(
                    frame["observations"][name],name=name,centered=centered,reset=reset)
                report["observation_reset"] = reset
                report["centered_observation"] = centered and name!="getup"
                if reset:
                    pending_reset = None
            out.write(json.dumps(report,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
