import json
from pathlib import Path
import pytest
from studio.config import RunConfig,command
from studio.registry import registry
from studio.recipe_training import skill_evaluation_command
from studio.jobs import Jobs


def config(rid,**changes):
    d=dict(next(r for r in registry()['recipes'] if r['id']==rid)['defaults'])
    d.update(changes)
    return RunConfig(**d)


@pytest.mark.parametrize('rid,kind',[('frontflip_native_ppo','frontflip'),('backflip_native_ppo','backflip'),('sideflip_native_ppo','sideflip')])
def test_form_seed_reaches_cpu_evaluator_but_never_trainer(tmp_path,rid,kind):
    motion=tmp_path/'artifacts/reference.npz';motion.parent.mkdir();motion.touch()
    options=dict(fixstand_roundtrip=True,evaluation_seed_start=25000)
    if rid=='sideflip_native_ppo':options['motion']='artifacts/reference.npz'
    if rid=='frontflip_native_ppo':options.update(flip_direction='front',delayed_entry=True)
    c=config(rid,recipe_options=options,auto_evaluate=True,seed=60)
    out=tmp_path/'outputs/studio/runs/PREVIEW'
    train=command(c,tmp_path,out,tmp_path/'override.json','/isaac/python.sh','/cpu/python')
    assert '--evaluation-seed-start' not in train
    evaluate=skill_evaluation_command(c,tmp_path,out,'/cpu/python')
    assert evaluate[0]=='/cpu/python'
    assert evaluate[evaluate.index('--kind')+1]==kind
    assert evaluate[evaluate.index('--seed-start')+1]=='25000'
    if kind=='sideflip':assert evaluate[evaluate.index('--motion')+1]==str(motion)
    assert skill_evaluation_command(c.model_copy(update={'auto_evaluate':False}),tmp_path,out,None) is None
    with pytest.raises(ValueError,match='training seed'):
        skill_evaluation_command(c.model_copy(update={'seed':25002}),tmp_path,out,'/cpu/python')


@pytest.mark.parametrize('invalid',[False,True])
def test_worker_waits_for_all_four_results_and_never_promotes_rejected_skill(tmp_path,monkeypatch,invalid):
    import studio.worker as worker
    c=config('frontflip_native_ppo',iterations=2,num_envs=32,auto_evaluate=True,
        recipe_options=dict(fixstand_roundtrip=True,flip_direction='front',delayed_entry=True,evaluation_seed_start=25000))
    jobs=Jobs(tmp_path);job=jobs.create(c.model_dump());launched=[]
    class Process:
        pid=12345;returncode=0
        def __init__(self,args,**kwargs):
            launched.append(args)
            if '--fixstand-output' in args:
                out=Path(args[args.index('--fixstand-output')+1]);out.mkdir(parents=True)
                (out/'model_1.pt').write_bytes(b'fake trainer output; not a real model')
                (out/'completion.json').write_text(json.dumps(dict(execution_complete=True,requested_iterations=2)))
            else:
                out=Path(args[args.index('--run')+1])
                episodes=[dict(seed=i,passed=False,motion_pass=False,constraints_pass=False) for i in range(25000,25004)]
                if invalid:episodes.pop()
                (out/'evaluation.json').write_text(json.dumps(dict(status='not_accepted',episodes=episodes)))
        def poll(self):return 0
    monkeypatch.setattr(worker.subprocess,'Popen',Process)
    if invalid:
        with pytest.raises(RuntimeError,match='Incomplete or invalid'):worker.execute(jobs,job,tmp_path,'/isaac/python.sh','/cpu/python')
    else:
        worker.execute(jobs,job,tmp_path,'/isaac/python.sh','/cpu/python')
        result=jobs.get(job['id'])
        assert result['state']=='succeeded' # Experiment executed, skill NOT accepted.
        assert result['detail']['simulation_acceptance']=='not_accepted'
        assert result['detail']['hardware_qualified'] is False
        assert len(result['detail']['evaluation'])==4
    assert len(launched)==2


def test_live_api_counts_only_declared_four_seed_range(tmp_path):
    from fastapi.testclient import TestClient
    from studio.api import create_app
    c=config('frontflip_native_ppo',recipe_options=dict(fixstand_roundtrip=True,
        delayed_entry=True,flip_direction='front',evaluation_seed_start=25000))
    jobs=Jobs(tmp_path);job=jobs.create(c.model_dump())
    jobs.update(job['id'],'evaluating',{'stage':'evaluation'})
    run=tmp_path/'outputs/studio/runs'/job['id']
    for seed in [24999,25000,25001,25004]:
        p=run/f'seed_{seed}/summary.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(successes=0,total=1)))
    with TestClient(create_app(tmp_path)) as client:
        result=client.get('/api/jobs').json()[0]
    assert result['evaluation_completed']==2
    assert result['evaluation_total']==4
