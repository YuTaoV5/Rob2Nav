"""Four separate uncut 12 s teacher-OFF clips with the evaluation command contract."""
import argparse,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,mediapy as media
from PIL import Image,ImageDraw
import rob2nav_expression as task
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
from rob2nav_contact_audit import BodyContactAudit
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=36230);a=p.parse_args()
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
with (out/'started.json').open('x') as f:json.dump({'checkpoint':a.checkpoint,'seed':a.seed},f)
for style,label in enumerate(task.STYLES):
 torch.manual_seed(a.seed);cfg=task.cfg(True);cfg.scene.num_envs=1;cfg.seed=a.seed
 cfg.episode_length_s=13.;cfg.viewer.width=854;cfg.viewer.height=480;cfg.viewer.distance=2.5;cfg.viewer.elevation=-20
 env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array')
 term=env.command_manager.get_term('target');original=term._resample_command
 def fixed(ids):
  original(ids);term.gen.style[ids]=float(style);term.gen.speed_cap[ids]=4.
  term.gen.chase_vel[ids]=task.velocity_command(term.gen.wait[ids],term.gen.style[ids],term.gen.speed_cap[ids])
 term._resample_command=fixed;obs,_=env.reset()
 ck=torch.load(a.checkpoint,map_location=env.device,weights_only=False);assert ck['cfg']['task']==task.TASK and not ck.get('initialization_only',False)
 actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).to(env.device);actor.load_state_dict(ck['actor']);actor.eval()
 body=BodyContactAudit(env);original_substep=env.metrics_manager.compute_substep
 def observe():original_substep();body.update()
 env.metrics_manager.compute_substep=observe;frames=[];trace=[]
 with torch.inference_mode():
  for i in range(round(12/env.step_dt)):
   command=term.gen.chase_vel.clone();obs,_,done,truncated,_=env.step(actor(obs['actor']))
   assert not bool((done|truncated).any()),'Recording terminated; retain failure, do not publish'
   trace.append({'t':(i+1)*env.step_dt,'command':command[0].tolist(),'phase_clock':float(term.gen.wait[0])})
   if i%2==0:
    im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,854,58),fill=(15,18,24))
    draw.text((10,8),f'SuperMimic expression student model800 | {label} | teacher OFF',fill='white')
    draw.text((10,30),f't={(i+1)*env.step_dt:.2f}s | authored rhythm; perceived emotion not validated',fill='white');frames.append(np.asarray(im))
 stem='expression_actorlow_v4_model800_'+label
 media.write_video(str(out/(stem+'.mp4')),frames,fps=25)
 meta={'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'style':label,'seed':a.seed,'frames':len(frames),'fps':25,'seconds':len(frames)/25,'continuous_unedited':True,'teacher_during_recording':False,'body_contact':body.summary(),'capability_qualified':False,'visual_review_complete':False,'emotion_perception_validated':False}
 (out/(stem+'.json')).write_text(json.dumps(meta,indent=2));(out/(stem+'_trace.json')).write_text(json.dumps(trace));env.close();print('VIDEO_COMPLETE',label,flush=True)
