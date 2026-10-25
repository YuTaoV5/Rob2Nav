"""Compare batched teacher inference/reset against the original batch-1 export."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rob2nav_moe_teacher import RoboGaugeHead
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
torch.manual_seed(34000)
head=RoboGaugeHead(args.policy,4,'cpu')
references=[torch.jit.load(args.policy,map_location='cpu').eval() for _ in range(4)]
for model in references:model.reset()
maximum=0.
with torch.inference_mode():
 for step in range(20):
  if step==7:
   head.reset(torch.tensor([False,False,True,False]));references[2].reset()
  obs=torch.randn(4,45)*.2
  batched=head(obs)
  expected=torch.cat([model(obs[i:i+1])[0] for i,model in enumerate(references)])
  maximum=max(maximum,float((batched-expected).abs().max()))
assert maximum<1e-5,maximum
report={'batch1_export_max_error':maximum,'independent_environment_reset_checked':True,'frames':20,'batch':4,'passed':True}
Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report))
