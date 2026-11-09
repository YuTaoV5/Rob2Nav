"""Run the complete flat and stair suites with only 41 sensor inputs per control update."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from train_student import BlindStudent, sensor_frame
from branch_teacher import BranchTeacher
from clearance import Clearance
from fit_branch_residual import Residual

sys.path.insert(0,str(Path("r157-flat-leveling").resolve()))
from as2w_contacts import exclude_nominal_overlaps
import evaluate_omni_stairs as ev
from evaluate_omni_joint import make_controller
from train_omni_stop_transition import load
from search_lateral_omni_teacher import course


@lru_cache(maxsize=1)
def model_for(checkpoint):
    saved=torch.load(checkpoint,weights_only=True,map_location="cpu")
    state=saved["actor"]
    model=BlindStudent(state["mean"],state["scale"]).eval()
    model.load_state_dict(state)
    return model,np.asarray(saved["default"])


@lru_cache(maxsize=1)
def flat_trim_for(path):
    from flat_trim_policy import network
    model=network().eval()
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["model"])
    return model


class Controller:
    def __init__(self, checkpoint):
        self.actor,self.defaults=model_for(checkpoint)
        self.state=None

    @torch.inference_mode()
    def action(self,obs):
        sensors=sensor_frame(obs,self.defaults)
        output,self.state=self.actor(torch.from_numpy(sensors)[None,None],self.state)
        return output[0,0].numpy()


class SharedPolicy:
    def __init__(self):
        self.stairs=Controller(checkpoint)
        self.flat=make_controller(policy,contract,build,base_factory())
        self.repeats=0
        self.pitch_repeats=0
        self.active_tick=None
        self.forward_start=None
        self.flat_steer_start=None
        self.flat_guard=None
        if globals().get("flat_guard_path"):
            from flat_contact_student import FlatContactStudent
            self.flat_guard=FlatContactStudent(flat_guard_path,geometry.inverse,flat_guard_threshold,
                                               flat_guard_gain,flat_guard_mix)
        self.clearance=self.make_clearance()
        self.residual=Residual(residual_path,self.stairs.defaults) if residual_path else None
        self.ppo=None
        if globals().get("ppo_path"):
            from stair_trim import LearnedTrim
            self.ppo=LearnedTrim(ppo_path,geometry.inverse,ppo_gain)

    def make_clearance(self):
        brake=None
        if globals().get("summit_model"):
            from summit_brake import SummitBrake
            brake=SummitBrake(summit_model,summit_stop)
        return Clearance(clearance_profile,geometry.inverse,contract["default_joint_pos"],brake)

    def action(self,obs,cmd,tick,oracle=None):
        if sensor32:
            obs=obs.copy()
            obs[3:9]=obs[3:9].astype(np.float32)
            defaults=self.stairs.defaults
            obs[10:26]=(obs[10:26]+defaults).astype(np.float32).astype(np.float64)-defaults
            obs[26:42]=obs[26:42].astype(np.float32)
        flat_cmd=cmd
        ramp=globals().get("flat_start_ramp",0)
        if ramp and abs(cmd[1])+abs(cmd[2])>.045:
            if self.flat_steer_start is None:
                self.flat_steer_start=tick
            t=np.clip((tick-self.flat_steer_start+1)/ramp,0.,1.)
            flat_cmd=cmd*(t*t*(3.-2.*t))
        else:
            self.flat_steer_start=None
        flat=self.flat.plan(None,None,obs,flat_cmd,tick,None)
        if globals().get("flat_trim_path") and abs(cmd[1])+abs(cmd[2])>.025:
            from flat_trim_policy import apply,features
            model=flat_trim_for(flat_trim_path)
            with torch.inference_mode():
                trim=model(torch.from_numpy(features(obs,flat))).numpy()
            flat=apply(obs,flat,trim)
        if self.flat_guard:
            flat=self.flat_guard.apply(obs,flat,cmd)
        self.last_flat=flat
        if cmd[0]<.13 or abs(cmd[1])+abs(cmd[2])>=.04:
            self.repeats=0
            self.pitch_repeats=0
            self.forward_start=None
            if self.active_tick is not None:
                self.active_tick=None
                self.stairs.state=None
                if self.residual:
                    self.residual.history.clear()
                self.clearance=self.make_clearance()
            return flat
        if self.forward_start is None:
            self.forward_start=tick
        if self.active_tick is None and oracle is None and tick-self.forward_start<start_lock:
            return flat
        if self.active_tick is None:
            shortfall,split,min_rear,n=gate_profile
            front=.086*obs[38:40].min()
            rear=.086*obs[40:42].mean()
            risk=cmd[0]-front>=shortfall and rear-front>=split and rear>=min_rear*cmd[0]
            self.repeats=self.repeats+1 if risk else 0
            pitch_contact=False
            profile=globals().get("pitch_gate")
            if profile is not None:
                tilt,rate,*delay=profile
                settled=tick-self.forward_start>=(delay[0] if delay else start_lock)
                rising=settled and -obs[6]>=tilt and -obs[4]>=rate and rear>=.45*cmd[0]
                self.pitch_repeats=self.pitch_repeats+1 if rising else 0
                pitch_contact=self.pitch_repeats>=2
            trigger=oracle if oracle is not None else self.repeats>=n or pitch_contact
            if trigger:
                self.active_tick=tick
        if self.active_tick is None:
            return flat
        stair=self.stairs.action(obs)
        x=min(1.,(tick-self.active_tick+1.)/8.)
        x=x*x*(3.-2.*x)
        action=flat+x*(stair-flat)
        age=tick-self.active_tick
        if self.residual:
            action=self.residual.apply(obs,action,age,residual_gain)
        action=self.clearance.apply(action,obs,age)
        if self.ppo and not self.clearance.hold:
            action=self.ppo(obs,action,age)
        return action


def evaluate(task):
    split, name, target, seed = task
    controller=SharedPolicy()
    first_latch=None
    first_hit=None
    peak_hit=None
    impacts=[]
    trace=[]
    tick=0
    peak_self=peak_ground=0.
    front_min=float("inf")
    force=np.zeros(6)
    video=None
    if video_cases:
        from video import Video
        stairs=name=="stairs"
        title="6 cm blind stairs | AS2-W student" if stairs else f"AS2-W | {name.replace('_',' ')} then stop"
        video=Video(out/f"{name}-{seed}.mp4",title,seed,stairs)

    def capture(model,data,command=None):
        nonlocal peak_self,peak_ground,front_min,first_hit,peak_hit,tick
        base=model.body("base_link").id
        wheels=[model.body(n).id for n in contract["joint_names"][12:]]
        pos=(data.xpos[wheels[:2]]-data.xpos[base])@data.xmat[base].reshape(3,3)
        front_min=min(front_min,pos[0,1],-pos[1,1])
        for i,c in enumerate(data.contact):
            a,b=(int(model.geom_bodyid[g]) for g in c.geom)
            if a and b:
                ev.sim.mj.mj_contactForce(model,data,i,force)
                peak_self=max(peak_self,force[0])
            elif (a or b) not in wheels:
                ev.sim.mj.mj_contactForce(model,data,i,force)
                if force[0]>=50.:
                    ground_geom=next((int(g) for g in c.geom if not model.geom_bodyid[g]),None)
                    row=dict(tick=tick,x=float(data.qpos[0]),body=model.body(a or b).name,
                             surface=model.geom(ground_geom).name if ground_geom is not None else None,
                             force=float(force[0]),phase="approach" if first_latch is None else "stairs")
                    if first_hit is None:
                        first_hit=row
                    impacts.append(row)
                    if force[0]>peak_ground:
                        peak_hit=row
                peak_ground=max(peak_ground,force[0])
        tick+=1
        if video:
            cmd=np.array((.4,0.,0.)) if command is None else command
            video.capture(model,data,cmd)

    if name=="stairs":
        model=ev.sim.load_flat_model(False,True,"stairs",seed,nominal_overlap_filter=True)
        stress=ev.sim.sample_dynamics_stress(seed,model.nbody-1,16)
        default=controller.stairs.defaults
        front={model.body(n).id for n in ("FL_foot","FR_foot")}
        surfaces={model.geom(f"stair_{i}").id for i in range(1,6)}
        command=np.array((.4,0.,0.))
        teacher=BranchTeacher(ev.sim,model,controller,default,limits,geometry,contract,branch_width) if branch_width else None
        frames=[]
        summit_samples=[]

        def plan(data,obs,elapsed,advance):
            nonlocal first_latch
            contact=None
            if use_oracle and split!="stairs-teacher-diagnostic":
                contact=False
                for i,c in enumerate(data.contact):
                    a,b=map(int,c.geom)
                    if b in surfaces:
                        a,b=b,a
                    if a in surfaces and model.geom_bodyid[b] in front:
                        ev.sim.mj.mj_contactForce(model,data,i,force)
                        contact |= force[0]>=8.
            current=round(elapsed/.02)
            nominal=controller.action(obs,command,current,contact)
            if summit_data and controller.active_tick is not None:
                top=model.geom("stair_5")
                front_edge=top.pos[0]-top.size[0]
                summit_samples.append((obs[:42].copy(),float(data.qpos[0]-front_edge),
                                       float(data.qvel[0]),controller.clearance.distance))
            action=teacher.choose(data,obs,current,advance,nominal) if teacher else nominal
            if teacher:
                frames.append((obs.copy(),nominal.copy(),action.copy(),controller.active_tick is not None))
            if first_latch is None:
                first_latch=controller.active_tick
            leg=np.clip(default[:12]+.35*action[:12],*limits[:12,:2].T)
            wheel=np.clip(16.*action[12:],-limits[12:,2],limits[12:,2])
            if trace_mode:
                trace.append(dict(obs=obs.tolist(),action=action.tolist(),qpos=data.qpos.tolist(),
                                  wheels=data.xpos[list(front)].tolist()))
            return action,leg,wheel

        row=ev.sim.evaluate(model,contract,dummy,seed,0.,limits=limits,action_clip=6.,
                            wheel_scale=16.,wheel_angle_period=4.*np.pi,dynamics_stress=stress,
                            terrain="stairs",observation_suffix=ev.gait_clock,native_planner=plan,
                            capture=lambda data:capture(model,data))
    else:
        spec=ev.sim.mj.MjSpec.from_file(str(assets/"local-step-diagnostic/as2w_flat.xml"))
        model=exclude_nominal_overlaps(spec).compile()
        model.geom_priority[model.geom("ground").id]=int(model.geom_priority.max())+1
        flat_teacher=None
        teacher_frames=[]
        if flat_teacher_mode:
            from flat_contact_teacher import FlatContactTeacher
            flat_teacher=FlatContactTeacher(model,geometry,contract)

        def plan(unused_model,unused_data,obs,command,tick,advance):
            nonlocal first_latch
            if split=="flat-transitions":
                command=np.asarray(target[min(tick//50,len(target)-1)])
                obs=obs.copy()
                obs[:3]=command-(.4,0.,0.)
            action=controller.action(obs,command,tick)
            if first_latch is None:
                first_latch=controller.active_tick
            if flat_teacher:
                selected=flat_teacher.choose(unused_data,advance,action,command)
                teacher_frames.append((obs.copy(),action.copy(),selected.copy()))
                return selected
            return action

        eval_target=np.zeros(3) if split=="flat-transitions" else np.asarray(target)
        row=ev.flat_episode(None,contract,limits,eval_target,seed,ev.gait_clock,
                            plan,capture,model=model)
        if split=="flat-transitions":
            row=dict(fallen=row["fallen"],passed=not row["fallen"] and first_latch is None)
        if flat_teacher:
            row.update(teacher_tests=flat_teacher.tests,teacher_changes=flat_teacher.changes)
            if teacher_frames:
                sensors,nominal,corrected=map(np.asarray,zip(*teacher_frames))
                np.savez_compressed(out/f"flat-{name}-{seed}.npz",sensors=sensors.astype(np.float32),
                                    nominal=nominal.astype(np.float32),corrected=corrected.astype(np.float32))
    if video:
        video.close()
    if branch_width and name=="stairs" and frames:
        end=first_hit["tick"] if first_hit is not None else len(frames)
        obs,nominal,action,active=map(np.asarray,zip(*frames))
        if active[:end].sum()>=25:
            np.savez_compressed(out/f"teacher-{seed}.npz",obs=obs[:end],nominal=nominal[:end],
                                action=action[:end],active=active[:end])
    if summit_data and name=="stairs" and summit_samples:
        sensors,offset,speed,wheel_distance=map(np.asarray,zip(*summit_samples))
        np.savez_compressed(out/f"summit-{seed}.npz",sensors=sensors.astype(np.float32),
                            offset=offset.astype(np.float32),speed=speed.astype(np.float32),
                            wheel_distance=wheel_distance.astype(np.float32))
    if trace_mode and trace:
        (out/f"trace-{seed}.json").write_text(json.dumps(dict(frames=trace,impacts=impacts)))
    return dict(row,split=split,command=name,seed=seed,
                qualified=bool(row["passed"] and front_min>0. and peak_self<1. and peak_ground<50.),
                front_min=float(front_min),peak_self=float(peak_self),peak_ground=float(peak_ground),
                latch_tick=first_latch,first_hit=first_hit,peak_hit=peak_hit,impacts=impacts,
                branch_changes=teacher.changes if name=="stairs" and teacher else 0)


if __name__=="__main__":
    assets,checkpoint,source,out,gate_path=map(Path,sys.argv[1:6])
    use_oracle="--oracle" in sys.argv
    trace_mode="--trace" in sys.argv
    sensor32="--float32" in sys.argv
    summit_data=os.environ.get("AS2W_COLLECT_SUMMIT")=="1"
    pitch_gate=tuple(map(float,os.environ["AS2W_PITCH_GATE"].split(","))) if "AS2W_PITCH_GATE" in os.environ else None
    summit_model=os.environ.get("AS2W_SUMMIT_MODEL")
    summit_stop=float(os.environ.get("AS2W_SUMMIT_STOP",".65"))
    if summit_model:
        sys.path.insert(0,str(Path("r174-summit-estimator").resolve()))
    branch_width=int(os.environ.get("AS2W_BRANCH_WIDTH","0"))
    flat_teacher_mode=os.environ.get("AS2W_FLAT_TEACHER")=="1"
    flat_guard_path=os.environ.get("AS2W_FLAT_GUARD")
    flat_guard_threshold=float(os.environ.get("AS2W_FLAT_GUARD_THRESHOLD",".5"))
    flat_guard_gain=float(os.environ.get("AS2W_FLAT_GUARD_GAIN",".8"))
    flat_guard_mix=os.environ.get("AS2W_FLAT_GUARD_MIX")=="1"
    if flat_teacher_mode or flat_guard_path:
        sys.path.insert(0,str(Path("r179-flat-contact-teacher").resolve()))
    flat_start_ramp=int(os.environ.get("AS2W_FLAT_START_RAMP","0"))
    flat_trim_path=os.environ.get("AS2W_FLAT_TRIM")
    if flat_trim_path:
        sys.path.insert(0,str(Path("r177-flat-feedback-training").resolve()))
    ppo_path=os.environ.get("AS2W_PPO_TRIM")
    ppo_gain=float(os.environ.get("AS2W_PPO_GAIN","1."))
    if ppo_path:
        sys.path.insert(0,str(Path("r171-contact-ppo").resolve()))
    residual_path=os.environ.get("AS2W_RESIDUAL")
    residual_gain=float(os.environ.get("AS2W_RESIDUAL_GAIN","1"))
    clearance_profile=tuple(map(float,os.environ.get("AS2W_CLEARANCE","0,0").split(",")))
    start_lock=int(os.environ.get("AS2W_START_LOCK_TICKS","0"))
    ev.sim.ROOT=assets
    policy,contract,limits,base_factory,build=load(assets,source)
    dummy=policy["anchor"]
    geometry=course(assets)[2]
    gate_profile=json.loads(gate_path.read_text())["profile"]
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(91291)
    cases=[("flat-fixed",name,target,seed) for name,target in ev.FLAT_COMMANDS.items() for seed in range(81201,81205)]
    cases.extend(("flat-independent",f"mixed_{i:02d}",target,91101+i)
                 for i,target in enumerate(rng.uniform((-.6,-.35,-.8),(.6,.35,.8),size=(48,3))))
    cases.extend(("flat-independent","stop",np.zeros(3),seed) for seed in range(91201,91205))
    cases.extend(("stairs-fixed","stairs",None,seed) for seed in range(27001,27017))
    cases.extend(("stairs-independent","stairs",None,seed) for seed in range(28001,28065))
    if "--flat-only" in sys.argv:
        cases=[case for case in cases if case[1]!="stairs"]
    elif "--full" not in sys.argv:
        cases=[case for case in cases if case[1]=="stairs"]
    if "--flat-development" in sys.argv:
        flat_seed=int(os.environ.get("AS2W_FLAT_SUITE_SEED","433100"))
        rng=np.random.default_rng(flat_seed)
        cases=[("flat-dev-fixed",name,target,flat_seed+100+i)
               for i,(name,target) in enumerate(ev.FLAT_COMMANDS.items())]
        cases.extend(("flat-dev-independent",f"mixed_{i:02d}",target,flat_seed+200+i)
                     for i,target in enumerate(rng.uniform((-.6,-.35,-.8),(.6,.35,.8),size=(32,3))))
    if trace_mode:
        cases=[case for case in cases if case[3] in (28003,28032,28041,27004,28005)]
    if os.environ.get("AS2W_SEEDS"):
        cases=[("stairs-experiment","stairs",None,int(seed)) for seed in os.environ["AS2W_SEEDS"].split(",")]
    if "--flat-transitions" in sys.argv:
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
    video_cases=os.environ.get("AS2W_VIDEO_CASES")
    if video_cases:
        sys.path.append(str(Path("r163-current-videos").resolve()))
        wanted=set(video_cases.split(","))
        cases=[case for case in cases if f"{case[1]}-{case[3]}" in wanted]
        assert len(cases)==len(wanted),(cases,wanted)
    workers=int(os.environ.get("AS2W_EVAL_WORKERS","6"))
    with ProcessPoolExecutor(workers) as pool, (out/"trials.jsonl").open("w",buffering=1) as stream:
        rows=[]
        for row in pool.map(evaluate,cases,chunksize=2):
            rows.append(row)
            stream.write(json.dumps(row,default=lambda x:x.item())+"\n")
            lengths=(16,52,80,104,120,184,len(cases))
            if len(rows) in lengths:
                batch=[r for r in rows if r["split"]==row["split"]]
                print(row["split"],len(batch),"standard pass",sum(r["passed"] for r in batch),
                      "qualified",sum(r["qualified"] for r in batch),
                      "latched",sum(r["latch_tick"] is not None for r in batch),"front side failures",
                      sum(r["front_min"]<=0. for r in batch),
                      "stairs qualified",Counter(r["stair_rise_m"] for r in batch if r["qualified"] and r["command"]=="stairs"),
                      flush=True)
