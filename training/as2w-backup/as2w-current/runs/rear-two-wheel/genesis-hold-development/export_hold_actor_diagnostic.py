"""Export the deterministic stage 1 development actor with its raw output."""
import argparse
from pathlib import Path
import onnx
import numpy as np
import torch
from torch import nn

parser=argparse.ArgumentParser()
parser.add_argument('checkpoint',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
weights=torch.load(args.checkpoint,weights_only=True,map_location='cpu')['actor_state_dict']
actor=nn.Sequential(nn.Linear(68,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,14))
actor.load_state_dict({key.removeprefix('mlp.'):value for key,value in weights.items() if key.startswith('mlp.')})
actor.eval()
example=torch.zeros(1,68)
torch.onnx.export(actor,example,args.output,input_names=['policy'],output_names=['raw_action'],
                  dynamic_axes={'policy':{0:'batch'},'raw_action':{0:'batch'}},opset_version=17,dynamo=False)
model=onnx.load(args.output)
onnx.checker.check_model(model)
from onnx.reference import ReferenceEvaluator
runtime=ReferenceEvaluator(model)
obs=torch.randn((64,68),generator=torch.Generator().manual_seed(45173))
with torch.inference_mode():
    predicted=actor(obs).numpy()
actual=runtime.run(None,{'policy':obs.numpy()})[0]
error=float(abs(predicted-actual).max())
print("REFERENCE_ABSOLUTE_GATE", error <= 1e-5)
print(f'ONNX_REFERENCE_MAX_ABS_ERROR {error:.8g}')

np.savez(args.output.with_suffix('.vectors.npz'),observations=obs.numpy(),raw_action=predicted)
