import sys,json,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,mediapy as media,numpy as np
from PIL import Image,ImageDraw
import rob2nav_velocity as v
from rob2nav_reset_targets import install
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic import state as st
parser=argparse.ArgumentParser(description="Record eight fixed velocity stages with explicit reset-hold control")
parser.add_argument("--checkpoint",required=True)
parser.add_argument("--output-dir",required=True)
parser.add_argument("--seed",type=int,default=32130)
parser.add_argument("--stage-seconds",type=float,default=4.)
args=parser.parse_args()
torch.manual_seed(args.seed)
p=Path(args.checkpoint)
ck=torch.load(p,map_location="cuda:0",weights_only=False)
cfg=install(v.flat_cfg(True));cfg.scene.num_envs=1;cfg.seed=args.seed
cfg.commands["target"].resampling_time_range=(10000.,10000.)
cfg.viewer.width=854;cfg.viewer.height=480;cfg.viewer.distance=3.;cfg.viewer.elevation=-22
env=ManagerBasedRlEnv(cfg=cfg,device="cuda:0",render_mode="rgb_array");obs,_=env.reset()
term=env.command_manager.get_term("target")
actor=Actor(obs["actor"].shape[-1],tuple(ck["cfg"]["actor_hidden"]),12).to(env.device)
actor.load_state_dict(ck["actor"]);actor.eval()
stages=[("STAND",(0.,0.,0.)),("FORWARD",(.6,0.,0.)),("BACKWARD",(-.3,0.,0.)),
 ("LEFT",(0.,.25,0.)),("RIGHT",(0.,-.25,0.)),("TURN LEFT",(0.,0.,.6)),
 ("TURN RIGHT",(0.,0.,-.6)),("STOP",(0.,0.,0.))]
command=torch.zeros((1,3),device=env.device)
original=term._resample_command
def fixed_resample(ids):
 original(ids);term.gen.chase_vel[ids]=command[ids]
term._resample_command=fixed_resample
frames=[];trace=[];falls=0;step=0;stage_seconds=args.stage_seconds
out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
with torch.inference_mode():
 for name,cmd in stages:
  command[:]=torch.tensor(cmd,device=env.device);term.gen.chase_vel.copy_(command)
  term._update_command(None);obs=env.observation_manager.compute(update_history=True)
  for i in range(round(stage_seconds/env.step_dt)):
   term.gen.chase_vel.copy_(command);a=actor(obs["actor"])
   obs,_,done,_,_=env.step(a);falls+=int(done.sum())
   state=st.robot_state(env)
   trace.append({"time":step*env.step_dt,"stage":name,"command":cmd,"velocity":state[:,st.V_YAW].cpu().tolist(),"yaw_rate":float(state[0,st.W_B][2]),"falls":falls})
   if step%2==0:
    image=Image.fromarray(env.render());draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,854,58),fill=(15,18,24))
    draw.text((10,8),f"SuperMimic | model 12000 + reset-hold | RESEARCH / NOT QUALIFIED",fill="white")
    draw.text((10,30),f"t={step*env.step_dt:.2f}s  {name}  cmd={cmd}  falls={falls}",fill="white")
    frames.append(np.asarray(image))
   step+=1
  print("RECORDED",name,step,flush=True)
fps=round(1/(2*env.step_dt))
media.write_video(str(out/"velocity_model12000_hold.mp4"),frames,fps=fps)
(out/"video_model12000_hold_trace.json").write_text(json.dumps(trace))
meta={"model_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"reset_control_sha256":hashlib.sha256(Path("rob2nav_reset_targets.py").read_bytes()).hexdigest(),
 "seed":args.seed,"frames":len(frames),"fps":fps,"duration_seconds":len(frames)/fps,"falls":falls,
 "initialize_hold_targets":True,"stages":stages,"stage_seconds":stage_seconds,"capability_qualified":False,
 "limitations":["one seed short demonstration","four-seed target rate failure remains","body contact smoke detected calf contact; full-speed visual review pending"]}
(out/"video_model12000_hold.json").write_text(json.dumps(meta,indent=2))
env.close();print("VIDEO_COMPLETED",json.dumps(meta),flush=True)

