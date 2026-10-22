"""Initialize a fresh optimizer run from the pinned upstream ONNX actor, with parity proof."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--critic-parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    pin='954aa61ceb62c515689677f44a21e32109f6b62d'
    assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==pin
    file=a.source/'model_6000_actor.onnx'
    tracked=subprocess.check_output(['git','-C',str(a.source),'rev-parse',pin+':model_6000_actor.onnx'],text=True).strip()
    assert subprocess.check_output(['git','-C',str(a.source),'hash-object',str(file)],text=True).strip()==tracked
    graph=onnx.load(file)
    assert [n.op_type for n in graph.graph.node]==['Gemm','Elu','Gemm','Elu','Gemm','Elu','Gemm']
    actor=torch.nn.Sequential(torch.nn.Linear(60,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),
        torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,12))
    state={v.name:torch.from_numpy(numpy_helper.to_array(v).copy()) for v in graph.graph.initializer}
    actor.load_state_dict(state,strict=True);actor.eval()
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(file),sess_options=options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(12100);obs=rng.normal(size=(128,60)).astype(np.float32)
    with torch.inference_mode():actual=actor(torch.from_numpy(obs)).numpy()
    expected=session.run(None,{session.get_inputs()[0].name:obs})[0]
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5)
    checkpoint=torch.load(a.critic_parent,map_location='cpu',weights_only=False)
    for name,value in state.items():
        assert checkpoint['model_state_dict']['actor.'+name].shape==value.shape
        checkpoint['model_state_dict']['actor.'+name]=value
    checkpoint['model_state_dict']['std'].fill_(.10)
    checkpoint['iter']=0
    provenance=dict(source_commit=pin,source_onnx_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        critic_parent_sha256=hashlib.sha256(a.critic_parent.read_bytes()).hexdigest(),
        parity_samples=128,max_absolute_error=float(abs(actual-expected).max()),actor_source='upstream model_6000_actor.onnx',
        critic_source='local v3 initialization only',optimizer_must_reset=True,hardware_qualified=False)
    checkpoint['infos']=dict(initialization=provenance)
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(checkpoint,a.output/'initialization.pt')
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)


if __name__=='__main__':main()
