import sys,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_velocity as v
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic import state as st
out=Path(os.environ.get("DIAG_OUT","reproduction/velocity/entry_diagnostic"));out.mkdir(parents=True,exist_ok=True)
result={}
for mode in ("policy","zero_action"):
 torch.manual_seed(int(os.environ.get("DIAG_SEED","32025")))
 cfg=v.flat_cfg(True);cfg.scene.num_envs=int(os.environ.get("DIAG_ENVS","1"));cfg.seed=int(os.environ.get("DIAG_SEED","32025"))
 cfg.commands["target"].resampling_time_range=(10000.,10000.)
 if os.environ.get("INIT_TARGET_HOLD")=="1":
  from rob2nav_reset_targets import install
  cfg=install(cfg)
 env=ManagerBasedRlEnv(cfg=cfg,device="cuda:0");obs,_=env.reset()
 robot=env.scene["robot"];term=env.command_manager.get_term("target");c=st.robot_consts(env)
 ck=torch.load(os.environ["DIAG_CHECKPOINT"],map_location=env.device,weights_only=False)
 actor=Actor(obs["actor"].shape[-1],tuple(ck["cfg"]["actor_hidden"]),12).to(env.device);actor.load_state_dict(ck["actor"]);actor.eval()
 term.gen.chase_vel.zero_();term._update_command(None);obs=env.observation_manager.compute(update_history=True)
 initial={"qpos":env.sim.data.qpos.cpu().tolist(),"qvel":env.sim.data.qvel.cpu().tolist(),"position_targets":robot.data.joint_pos_target.cpu().tolist()}
 trace=[];original=env.metrics_manager.compute_substep
 def sample():
  original()
  trace.append({"q":robot.data.joint_pos.cpu().tolist(),"qd":robot.data.joint_vel.cpu().tolist(),
   "tau":env.sim.data.qfrc_actuator[:,robot.indexing.joint_v_adr].cpu().tolist(),"a":env.action_manager.action.cpu().tolist()})
 env.metrics_manager.compute_substep=sample
 with torch.inference_mode():
  for _ in range(10):
   term.gen.chase_vel.zero_()
   a=actor(obs["actor"]) if mode=="policy" else torch.zeros((cfg.scene.num_envs,12),device=env.device)
   obs,_,_,_,_=env.step(a)
 result[mode]={"initial":initial,"trace":trace,"joint_names":env.action_manager.get_term("joint_pos")._target_names,"default_q":c.default_q.cpu().tolist(),"action_scale":c.action_scale.cpu().tolist()}
 env.close()
summary={"same_initial_state":result["policy"]["initial"]==result["zero_action"]["initial"]}
for mode,d in result.items():
 peaks=[(abs(x)/(20.07 if "calf" in d["joint_names"][j] else 30.1),abs(x),i,e,j) for i,row in enumerate(d["trace"]) for e,qd in enumerate(row["qd"]) for j,x in enumerate(qd)]
 ratio,value,i,e,j=max(peaks);summary[mode]={"peak_ratio":ratio,"environment":e,"peak_abs_velocity":value,"end_time":(i+1)*.005,"joint":d["joint_names"][j],"first_action":d["trace"][0]["a"][0]}
(out/"trace.json").write_text(json.dumps(result));(out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

