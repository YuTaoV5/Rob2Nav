"""Submit one bounded full-episode check only after the existing v2 smoke is audited.

Does not start long training or promote a model. Creation is never blindly retried.
"""
from pathlib import Path
import copy,fcntl,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from studio.pipeline_api import request
JOB='f78810286cd54adf'
NAME='backflip_landing_v2_episode_check_20261024'


def prepare(config):
    c=copy.deepcopy(config)
    c.update(name=NAME,num_envs=32,iterations=24,seed=74,auto_evaluate=True)
    c['recipe_options']['evaluation_seed_start']=28800
    return c


def verify(run):
    import yaml
    from rob2nav.export_actor import ConfigLoader
    from skills.backflip.landing_contract import validate_landing_contract
    completion=json.loads((run/'completion.json').read_text())
    if not completion['execution_complete'] or completion['requested_iterations']!=2:
        raise ValueError('Expected completed two-update configuration smoke')
    env=yaml.load((run/'params/env.yaml').read_text(),Loader=ConfigLoader)
    contract=json.loads((run/'fixstand.json').read_text())
    audited=validate_landing_contract(env,contract)
    if audited['settings']['cycles']!=3 or audited['target_rate_objective']!={'weight':5.,'margin':.8,'space':'mapped_clipped_sdk_actions'}:
        raise ValueError('Smoke configuration differs from planned experiment')
    export=json.loads((run/'exported/export.json').read_text())
    if export['rebound_contract']!=audited:raise ValueError('Actual export contract mismatch')
    policy=hashlib.sha256((run/'exported/policy.onnx').read_bytes()).hexdigest()
    if policy!=export['policy_sha256']:raise ValueError('Policy hash mismatch')
    result=json.loads((run/'evaluation.json').read_text())
    if result['status'] not in ('not_accepted','passed_simulation_only'):
        raise ValueError('Four-seed chain has not completed')
    if [e['seed'] for e in result['episodes']]!=list(range(28700,28704)):
        raise ValueError('Wrong independent seeds')
    for seed in range(28700,28704):
        s=json.loads((run/f'seed_{seed}/summary.json').read_text())
        if (s.get('trained_execution_contract') is not True or s['policy_sha256']!=policy or
            s['metrics']['execution_contract']!=audited or not s['metrics']['rebound_motion']['coverage_complete']):
            raise ValueError('Incomplete or mismatched physical evaluation')
    return dict(actual_contract=audited,evaluation_status=result['status'],policy_sha256=policy,
                scope='runtime/configuration checked, motion qualification is not required for a bounded episode probe')


def read_live_state(path, attempts=5):
    """Retry only transient partial JSON reads; never retry job creation."""
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text()) if path.exists() else {}
        except json.JSONDecodeError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2)


def main():
    base=ROOT/'outputs/landing_episode_queue_20261024';base.mkdir(exist_ok=True)
    lock=(base/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def record(**kw):(base/'state.json').write_text(json.dumps(dict(job=JOB,updated=time.time(),**kw),indent=2))
    try:
        if (base/'create_attempt.json').exists():raise RuntimeError('Prior creation attempt exists; inspect before any retry')
        while True:
            jobs=request('jobs');jobs=jobs if isinstance(jobs,list) else jobs['jobs']
            existing=[j for j in jobs if j['name']==NAME]
            if existing:
                record(stage='existing_task',ids=[j['id'] for j in existing]);return
            job=next(j for j in jobs if j['id']==JOB)
            if job['state'] not in ('draft','queued','preflight','running','succeeded'):
                raise RuntimeError('Configuration smoke terminated: '+job['state'])
            wait_state=ROOT/'outputs/landing_after_smoke_20261024/state.json'
            evaluation=read_live_state(wait_state)
            if evaluation.get('stage')=='failed':raise RuntimeError('Existing evaluator failed: '+str(evaluation.get('error')))
            if job['state']=='succeeded' and evaluation.get('stage')=='completed':break
            record(stage='waiting',training_state=job['state'],evaluation_stage=evaluation.get('stage'))
            time.sleep(30)
        verified=verify(ROOT/'outputs/studio/runs'/JOB)
        (base/'verified.json').write_text(json.dumps(verified,indent=2))
        config=prepare(job['config']);check=request('preflight',config)
        (base/'preflight.json').write_text(json.dumps(check,indent=2))
        if not check['valid']:raise ValueError(check['errors'])
        (base/'create_attempt.json').write_text(json.dumps(config,indent=2))
        created=request('jobs',config)
        (base/'created.json').write_text(json.dumps(created,indent=2))
        submitted=request('jobs/'+created['id']+'/submit',{})
        (base/'submitted.json').write_text(json.dumps(submitted,indent=2))
        record(stage='submitted',next_job=created['id'],capability_verified=False)
    except Exception as exc:
        record(stage='failed',error=str(exc));raise


if __name__=='__main__':main()
