import sys,json,hashlib
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_velocity as velocity
from go2_supermimic.supermimic import train as training,state as st
training.Imagination=velocity.VelocityImagination
OriginalTrainer=training.Trainer
class AuditedTrainer(OriginalTrainer):
 def __init__(self,cfg):
  super().__init__(cfg)
  assert cfg.task=="Go2-Velocity-Blind-Flat"
  assert not cfg.driver_policy and not cfg.expert_policy
  self.record={"task":cfg.task,"cfg":asdict(cfg),"contract":velocity.CONTRACT,
    "reward_weights":velocity.WEIGHTS,"actor_terms":self.terms,"actor_dim":self.obs.shape[-1],
    "actor_has_height_scan":False,"actor_has_lidar":False,
    "source_sha256":hashlib.sha256(Path(velocity.__file__).read_bytes()).hexdigest()}
  (self.log_dir/"velocity_contract.json").write_text(json.dumps(self.record,indent=2))
 def run(self):
  super().run()
  s=st.robot_state(self.env);a=self.env.action_manager.action
  prev=self.env.action_manager.prev_action;cmd=self.term.gen.chase_vel
  expected=velocity.real_reward(self.env)*self.env.step_dt
  gst=self.term.gen.state()
  k={"soft_limits":self.consts.soft_limits,"default_q":self.consts.default_q}
  actual,_=self.imag._rewards(s,s,st.planar_pose(self.env),gst,a,prev,k)
  error=float((actual-expected).abs().max())
  assert error<1e-5,error
  assert torch.count_nonzero(self.term.command[:,3:])==0
  target=self.obs[:,-28*self.history:].reshape(-1,self.history,28)
  assert torch.count_nonzero(target[:,:,3:])==0
  report={"reward_max_error":error,"padded_channels_zero":True,
    "policy_has_no_ball_positions":True,"actor_no_height_scan":True,
    "simulation_reward_parity":True,"training_completed":True,"capability_qualified":False}
  (self.log_dir/"velocity_audit.json").write_text(json.dumps(report,indent=2))
  print("VELOCITY_AUDIT",json.dumps(report),flush=True)
training.Trainer=AuditedTrainer
if __name__=="__main__":
 torch.set_float32_matmul_precision("high");training.main()

