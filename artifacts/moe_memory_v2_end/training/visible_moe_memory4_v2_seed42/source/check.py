"""Multi-stream history/reset equivalence, trainability and mask checks."""
import argparse,hashlib,json
from pathlib import Path
import torch
from .model import load_base,PerceptiveActor,ACTOR_DIM
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(711)
base=load_base(a.checkpoint).eval();refs=[torch.jit.load(a.checkpoint).eval() for _ in range(4)]
history=torch.zeros(4,5,45);maximum=0.
with torch.no_grad():
    for step in range(64):
        if step in (17,42):history[1]=0;refs[1].reset()
        obs=torch.randn(4,45)*.2
        history=torch.cat((history[:,1:],obs[:,None]),1)
        reference=torch.cat([r(obs[i:i+1])[0] for i,r in enumerate(refs)])
        result=base(history.flatten(1))
        maximum=max(maximum,float((result-reference).abs().max()))
        # Batched GEMM and the single-stream scripted kernels round differently.
        torch.testing.assert_close(result,reference,atol=2e-5,rtol=2e-5)
actor=PerceptiveActor(a.checkpoint)
x=torch.randn(4,ACTOR_DIM)*.1;x[:,225+693:225+2*693]=0
torch.testing.assert_close(actor(x),actor.base(x[:,:225]),atol=0,rtol=0)
assert torch.isfinite(actor(x)).all()
x[:,225+693:225+2*693]=1
frozen={k:v.clone() for k,v in actor.base.state_dict().items()}
opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=1e-3)
for _ in range(2):
    opt.zero_grad();loss=(actor(x)-torch.ones(4,12)).square().mean();loss.backward();opt.step()
assert actor.fusion[-1].weight.grad.abs().sum()>0
assert actor.terrain.cnn[0].weight.grad.abs().sum()>0
for k,v in frozen.items():assert torch.equal(v,actor.base.state_dict()[k])
actor.set_phase('actor');assert actor.base.actor[0].weight.requires_grad
assert not actor.base.student_moe_encoder.experts_out.weight.requires_grad
actor.set_phase('joint');assert actor.base.student_moe_encoder.experts_out.weight.requires_grad
assert not any(p.requires_grad for p in actor.anchor.parameters())
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
report={'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        'streams':4,'steps':64,'partial_resets':[17,42],'max_action_error':maximum,
        'zero_fusion_exact':True,'all_unknown_finite':True,'fusion_trainable':True,
        'base_frozen_in_fusion_phase':True,'staged_unfreezing_verified':True}
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
