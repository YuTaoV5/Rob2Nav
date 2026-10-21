"""Serial GPU jobs for standard-stance action round trips, with durable stage status."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=500);p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args();root=args.root.resolve();out=args.output.resolve()
    if not 1<=args.iterations<=5000 or not 8<=args.num_envs<=4096:raise ValueError('Invalid training budget')
    out.mkdir(parents=True,exist_ok=False)
    parent=root.parent;isaac=Path('/root/isaacsim/python.sh');cpu=parent/'envs/go2-mapless/bin/python'
    source=parent/'Rob2Nav-upstreams/GO2_backflip';unitree=parent/'Rob2Nav-upstreams/unitree_rl_lab'
    side=root/'artifacts/release_20261021/weights/sideflip/selected/model_5200.pt'
    upright=root/'outputs/upright_walk_full_20261021/model_4999.pt'
    back=root/'artifacts/release_20261021/weights/backflip/selected/model_4999.pt'
    for path in (isaac,cpu,source,unitree,side,upright,back):
        if not path.exists():raise FileNotFoundError(path)
    state=dict(started=time.time(),status='running',budget=vars(args)|{'root':str(root),'output':str(out)},stages=[],
               checkpoint_sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (side,upright,back)},
               source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'skills').glob('*fixstand*.py')},
               hardware_connected=False,independently_validated=False)
    def save():
        temp=out/'status.tmp';temp.write_text(json.dumps(state,default=str,indent=2)+'\n');temp.replace(out/'status.json')
    def run(name,cmd,required):
        inputs=list((root/'skills').glob('*fixstand*.py'))+[root/f for f in ('skills/sideflip/train.py','skills/sideflip/task.py',
            'skills/sideflip/common.py','skills/sideflip/build_motion.py','skills/upright_walk/train.py','skills/upright_walk/task.py','skills/backflip/train_isaac51.py')]
        stage=dict(name=name,command=list(map(str,cmd)),started=time.time(),status='running',
                   source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs})
        state['stages'].append(stage);save()
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(list(map(str,cmd)),cwd=root,stdout=log,stderr=subprocess.STDOUT)
        stage.update(exit_code=result.returncode,finished=time.time())
        if result.returncode or any(not path.exists() for path in required):
            stage['status']='failed';state['status']='failed';save();raise RuntimeError(name+' failed; inspect its log')
        stage['status']='completed';save()
    try:
        csv=out/'sideflip.csv';motion=out/'sideflip.npz'
        run('reference_csv',[cpu,'-m','skills.fixstand_reference','--csv',parent/'capability-work/sideflip/input.csv','--output',csv],[csv])
        run('reference_fk',[isaac,'-m','skills.sideflip.build_motion','--backflip-source',source,'--csv',csv,'--output',motion,'--headless','--device','cuda:0'],[motion,motion.with_suffix('.json')])
        common=['--num-envs',args.num_envs,'--iterations',args.iterations,'--headless','--device','cuda:0','--transfer-motors','--fixstand-roundtrip']
        run('sideflip',[isaac,'-m','skills.sideflip.train','--backflip-source',source,'--unitree-source',unitree,'--motion',motion,'--resume',side,'--output',out/'sideflip',*common],
            [out/'sideflip/completion.json'])
        run('upright',[isaac,'-m','skills.upright_walk.train','--backflip-source',source,'--unitree-source',unitree,'--resume',upright,'--output',out/'upright','--reset-std','.25',*common],
            [out/'upright/completion.json'])
        run('backflip',[isaac,'-m','skills.backflip.train_isaac51','--source',source,'--fixstand-roundtrip','--fixstand-output',out/'backflip',
            '--resume','--checkpoint',back,'--reset_optimizer','--num_envs',args.num_envs,'--max_iterations',args.iterations,
            '--trainer','rsl_rl','--headless','--device','cuda:0'],[out/'backflip/fixstand.json'])
        for name in ('sideflip','upright','backflip'):
            models=list((out/name).glob('model_*.pt'))
            if not models:raise RuntimeError(name+' did not produce a checkpoint')
        state['status']='training_completed_not_validated';state['finished']=time.time();save()
    except BaseException as exc:
        state.update(status='failed',reason=repr(exc));save();raise


if __name__=='__main__':main()
