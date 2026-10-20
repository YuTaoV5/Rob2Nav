"""CPU-only audited upstream layer check; no nav weights or robot dynamics."""
import argparse,json
from pathlib import Path
import torch
from rob2nav.sea_contract import load_native_cbf,validate_geometry,SOURCE_COMMIT,SOURCE_SHA256

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(802)
layer=load_native_cbf(args.source)
u=torch.randn(16,3,requires_grad=True)
ranges=torch.full((16,41),3.);ranges[:,0]=.22
alpha=torch.ones(16,1,requires_grad=True)
out=layer(u,ranges,alpha)
assert torch.isfinite(out).all()
torch.testing.assert_close(out[:,2],u[:,2])
out.square().sum().backward()
assert torch.isfinite(u.grad).all() and torch.isfinite(alpha.grad).all()
legacy=load_native_cbf(args.source,fov_deg=180)
legacy_output=legacy(u.detach(),ranges,alpha.detach())
try:validate_geometry(legacy)
except ValueError:rejected=True
else:raise AssertionError('180-degree layer accepted for 240-degree observations')
# A checkpoint can silently replace registered buffers: check again after load.
layer.load_state_dict(legacy.state_dict())
try:validate_geometry(layer)
except ValueError:buffer_rejected=True
else:raise AssertionError('Mismatched restored ray buffer accepted')
record=dict(source_commit=SOURCE_COMMIT,source_sha256=SOURCE_SHA256,device='cpu',seed=802,
    batch=16,rays=41,fov_deg=240,finite_forward=True,finite_gradients=True,yaw_passthrough=True,
    legacy_fov_rejected=rejected,restored_mismatch_rejected=buffer_rejected,
    max_command_delta_180_vs_240=float((out.detach()-legacy_output).abs().max()),
    scope='Isolated native CBF layer only. No trained SEA policy, robot dynamics or safety guarantee.')
path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2));print(json.dumps(record))
