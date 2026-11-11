"""Run the server's benchmark using either the ONNX policy or paired replay."""
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from export_r152_onnx import session_for
from r177_onnx_model import STATE_SIZE, FLAT_STATE

import evaluate_handoff as e

reference=e.SharedPolicy
assets,package,out=map(Path,sys.argv[1:4])
weights=package/"weights"
model=package/"unitree_cpp_deploy/logs/as2w/r152/exported/policy.onnx"
fixture=[]
active_frames=0
difference=0.
mode_errors=0


@lru_cache(maxsize=1)
def session():
    return session_for(model)


class Actor:
    def __init__(self):
        self.stairs=SimpleNamespace(defaults=defaults)
        self.active_tick=None
        self.memory=np.zeros((1,STATE_SIZE),dtype=np.float32)

    def action(self,obs,cmd,tick,oracle=None):
        frame=np.r_[cmd,obs[3:9],obs[10:26]+defaults,obs[26:42]].astype(np.float32)
        action,self.memory=session().run(["actions","next_state"],
                                         {"obs":frame[None],"state":self.memory})
        age=self.memory[0,FLAT_STATE+3]
        self.active_tick=int(tick+1-age) if age>0 else None
        if "--parity" in sys.argv:
            fixture.append((episode,*frame.tolist(),*action[0].tolist()))
        return action[0]


class Paired:
    def __init__(self):
        self.python=reference()
        self.onnx=Actor()
        self.stairs=self.python.stairs

    @property
    def active_tick(self):
        return self.python.active_tick

    def action(self,obs,cmd,tick,oracle=None):
        global difference,active_frames,mode_errors
        expected=self.python.action(obs,cmd,tick,oracle)
        actual=self.onnx.action(obs,cmd,tick,oracle)
        difference=max(difference,float(np.max(np.abs(actual-expected))))
        active_frames+=self.active_tick is not None
        mode_errors+=self.active_tick!=self.onnx.active_tick
        return expected


def setup():
    torch.set_num_threads(1)
    e.ev.sim.ROOT=e.assets=assets
    e.checkpoint=weights/"stairs-gru.pt"
    e.policy,e.contract,e.limits,e.base_factory,e.build=e.load(assets,weights/"flat-policy.pt")
    e.dummy=e.policy["anchor"]
    e.geometry=e.course(assets)[2]
    e.gate_profile=json.loads((weights/"wheel-handoff.json").read_text())["profile"]
    e.clearance_profile=(0.,1.95)
    e.start_lock=25
    e.pitch_gate=(.10,.50)
    e.residual_path=weights/"stairs-residual.pt"
    e.residual_gain=.8
    e.ppo_path=weights/"stairs-ppo.pt"
    e.ppo_gain=1.
    e.flat_trim_path=weights/"flat-ppo.pt"
    e.trace_mode=e.use_oracle=e.summit_data=e.flat_teacher_mode=e.video_cases=False
    e.branch_width=0
    e.sensor32=True
    return np.asarray(e.contract["default_joint_pos"])


if __name__=="__main__":
    defaults=setup()
    out.mkdir(exist_ok=True,parents=True)
    if "--parity" in sys.argv:
        e.SharedPolicy=Paired
        cases=[
            ("flat-transitions","transitions",((.4,0.,0.),(.35,.25,0.),(.25,-.2,.45),(0.,0.,-.55),(0.,0.,0.)),427903),
            ("stairs-fixed","stairs",None,27012),
            ("stairs-fixed","stairs",None,27015),
        ]
        for episode,case in enumerate(cases,1):
            e.evaluate(case)
        with (out/"r177-trace.csv").open("w") as f:
            csv.writer(f,lineterminator="\n").writerows(fixture)
        results=dict(frames=len(fixture),stair_mode_frames=active_frames,mode_errors=mode_errors,
                     max_abs_action_error=difference)
        (out/"parity.json").write_text(json.dumps(results,indent=2))
        print(results,flush=True)
        assert difference<.001 and mode_errors==0,results
    elif "--video" in sys.argv:
        e.SharedPolicy=Actor
        e.video_cases=True
        e.out=out
        cases=[("flat-fixed",name,e.ev.FLAT_COMMANDS[name],seed) for name,seed in
               (("forward",81201),("reverse",81201),("left",81203),("arc_right",81204),("turn_right",81201))]
        cases.append(("stairs-independent","stairs",None,28029))
        with (out/"rollouts.jsonl").open("w",buffering=1) as f:
            for case in cases:
                row=e.evaluate(case)
                assert row["qualified"],(case,row["peak_ground"])
                f.write(json.dumps(row,default=lambda x:x.item())+"\n")
    else:
        e.SharedPolicy=Actor
        rng=np.random.default_rng(91291)
        cases=[("flat-fixed",name,target,seed) for name,target in e.ev.FLAT_COMMANDS.items() for seed in range(81201,81205)]
        cases.extend(("flat-independent",f"mixed_{i:02d}",target,91101+i)
                     for i,target in enumerate(rng.uniform((-.6,-.35,-.8),(.6,.35,.8),size=(48,3))))
        cases.extend(("flat-independent","stop",np.zeros(3),seed) for seed in range(91201,91205))
        cases.extend(("stairs-fixed","stairs",None,seed) for seed in range(27001,27017))
        cases.extend(("stairs-independent","stairs",None,seed) for seed in range(28001,28065))
        name="onnx-standard.jsonl"
        if "--transitions" in sys.argv:
            rng=np.random.default_rng(427901)
            stop=(0.,0.,0.)
            cases=[]
            for i in range(64):
                yaw=.55*(-1)**i
                side=.26*(-1)**(i//2)
                forward=(rng.uniform(.20,.60),0.,0.)
                next_forward=(rng.uniform(.16,.52),0.,0.)
                profiles=(
                    (stop,forward,stop,next_forward,stop,stop),
                    ((0.,0.,yaw),forward,(0.,side,0.),next_forward,stop,stop),
                    ((-.42,0.,0.),forward,(-.32,side,yaw),next_forward,stop,stop),
                    ((.35,side,yaw),forward,(.3,0.,-yaw),next_forward,stop,stop),
                )
                cases.append(("flat-transitions",f"profile-{i%4}",profiles[i%4],427900+i))
            name="onnx-transitions.jsonl"
        with ProcessPoolExecutor(8) as pool,(out/name).open("w",buffering=1) as f:
            for row in pool.map(e.evaluate,cases,chunksize=2):
                f.write(json.dumps(row,default=lambda x:x.item())+"\n")
