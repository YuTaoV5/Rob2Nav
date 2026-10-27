"""Mechanism tests only: schedule, coordinate invariance, reward and gradients."""
from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_animal as base
import rob2nav_animal_velocity_objective as proposal
from rob2nav_kine_reference import yaw_quaternion
from go2_supermimic.supermimic import state as st
ref,_,_,_=base.parameters('cpu');length=ref.length
phase=torch.tensor([0,1,2,10,length-1,length,length+1,2*length-1,2*length,3*length+4],dtype=torch.float32)
target=proposal.reference_velocity(phase);heading=yaw_quaternion(torch.linspace(-2.5,2.5,len(phase)));rotated=proposal.reference_velocity(phase,heading)
rotation_error=float((target-rotated).abs().max());assert rotation_error<1e-4 and torch.isfinite(target).all()
q,z,g,_=base.desired(phase);a=torch.zeros((len(phase),12));state=torch.zeros((len(phase),st.STATE_DIM));state[:,st.Q]=q;state[:,st.Z]=z;state[:,st.GRAV]=g
moving=state.clone();moving[:,st.V_YAW][:,:2]=target
old0=base.reward(state,phase,a,a,a,torch.full_like(a,.25));old1=base.reward(moving,phase,a,a,a,torch.full_like(a,.25));assert torch.equal(old0,old1)
new0=proposal.combined_reward(state,phase,a,a,a,torch.full_like(a,.25));new1=proposal.combined_reward(moving,phase,a,a,a,torch.full_like(a,.25));assert (new1>=new0).all()
probe=state.clone().requires_grad_(True);proposal.velocity_bonus(probe,phase).sum().backward();gn=float(probe.grad[:,st.V_YAW][:,:2].norm());assert torch.isfinite(probe.grad).all() and gn>0
report={'candidate_only':True,'training_started':False,'weight':2.,'sigma_mps':.25,'motion_length':length,'tested_frames':phase.tolist(),'reference_yaw_velocity':target.tolist(),'world_heading_invariance_max_error':rotation_error,'old_reward_stationary_vs_target_max_difference':float((old1-old0).abs().max()),'new_reward_target_minus_stationary_mean':float((new1-new0).mean()),'velocity_state_gradient_norm_at_stationary':gn,'finite_gradient':True,'actor_command_schema_changed':False,'actual_velocity_exposed_to_actor':False,'world_model_action_gradient_calibrated':False,'limitations':'Scalar mechanism checks do not validate WM action response, policy learning or improved motion.'}
report['source_sha256']={Path(proposal.__file__).name:hashlib.sha256(Path(proposal.__file__).read_bytes()).hexdigest(),Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out=Path(__file__).resolve().parents[1]/'reproduction/animal_expression_v1/animal_velocity_objective_audit.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
