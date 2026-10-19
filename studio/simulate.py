"""Native checkpoint inference wrapper. Inspect metadata in Isaac's Python only."""
import argparse
import subprocess
import sys
import torch

def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','kind','course','difficulty','seed','max-seconds','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if not isinstance(checkpoint,dict) or 'model_state_dict' not in checkpoint:raise ValueError('Expected a MoE perceptive training checkpoint; TorchScript/base/NJU formats are not interchangeable.')
    source=checkpoint['infos']['provenance']['args']
    args=[sys.executable,'-m','moe_perceptive.evaluate','--headless','--num-envs','1','--video','--studio-stream',
        '--moe','vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt']
    for key,value in vars(a).items():args.extend(['--'+key.replace('_','-'),str(value)])
    for flag in ('navigation','teacher_map'):
        if source.get(flag):args.append('--'+flag.replace('_','-'))
    args.extend(['--sensor-rig',source.get('sensor_rig','original'),'--memory-lifetime',str(source.get('memory_lifetime',1.))])
    result=subprocess.run(args)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
