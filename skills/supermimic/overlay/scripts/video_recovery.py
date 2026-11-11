import argparse,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,mediapy as media
from PIL import Image,ImageDraw
import rob2nav_recovery as task
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=35130);args=p.parse_args();torch.manual_seed(args.seed)
cfg=task.cfg(True);cfg.scene.num_envs=1;cfg.seed=args.seed;cfg.episode_length_s=13.;cfg.viewer.width=854;cfg.viewer.height=480;cfg.viewer.distance=2.5;cfg.viewer.elevation=-20
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array');obs,_=env.reset();ck=torch.load(args.checkpoint,map_location=env.device,weights_only=False);assert ck['cfg']['task']==task.TASK;actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).to(env.device);actor.load_state_dict(ck['actor']);actor.eval();frames=[];trace=[]
with torch.inference_mode():
 for i in range(600):
  obs,_,done,trunc,_=env.step(actor(obs['actor']));assert not bool((done|trunc).any());s=task.st.robot_state(env);trace.append({'t':(i+1)*.02,'height':float(s[0,task.st.Z][0]),'upright':float(-s[0,task.st.GRAV][2])})
  if i%2==0:
   image=Image.fromarray(env.render());draw=ImageDraw.Draw(image);draw.rectangle((0,0,854,58),fill=(15,18,24));draw.text((10,8),'SuperMimic recovery student | model4000 | teacher OFF',fill='white');draw.text((10,30),f't={(i+1)*.02:.2f}s seed={args.seed} | research, not hardware qualified',fill='white');frames.append(np.asarray(image))
out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);stem='recovery_student_v1_model4000';media.write_video(str(out/(stem+'.mp4')),frames,fps=25);(out/(stem+'_trace.json')).write_text(json.dumps(trace));meta={'model_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),'seed':args.seed,'frames':len(frames),'fps':25,'duration_seconds':len(frames)/25,'teacher_during_recording':False,'continuous_unedited':True,'normal_reset_only_before_recording':True,'task':task.TASK,'capability_qualified':False,'hardware_qualified':False};(out/(stem+'.json')).write_text(json.dumps(meta,indent=2));print('RECOVERY_VIDEO',meta,flush=True);env.close()
