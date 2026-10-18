import json
from pathlib import Path
import zipfile
import pytest
from fastapi.testclient import TestClient
from studio.api import create_app
from studio.catalog import Catalog,normalize_evaluation
from studio.config import RunConfig,command,resolve_checkpoint,simulation_command
from studio.jobs import Jobs
from studio.worker import checked_training
from moe_perceptive.studio_overrides import load,apply_rewards


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data),encoding='utf-8')


def test_running_job_progress_and_evaluation_counter(tmp_path):
    jobs=Jobs(tmp_path);j=jobs.create(RunConfig(name='Progress test').model_dump())
    jobs.update(j['id'],'running',{'stage':'training'})
    write(tmp_path/'outputs/studio/runs'/j['id']/'status.json',{'iteration':71,'phase':'joint','steps_per_s':3000})
    with TestClient(create_app(tmp_path)) as c:
        item=c.get('/api/jobs').json()[0]
        assert item['progress']['iteration']==71 and item['progress_updated']>0
        jobs.update(j['id'],'evaluating',{'stage':'evaluation'})
        write(tmp_path/'outputs/studio/jobs'/j['id']/'evaluation/flat/summary.json',{'total':32})
        bad=tmp_path/'outputs/studio/jobs'/j['id']/'evaluation/stairs/summary.json'
        bad.parent.mkdir(parents=True);bad.write_text('{')
        assert c.get('/api/jobs').json()[0]['evaluation_completed']==1


def test_gpu_compute_process_telemetry(tmp_path,monkeypatch):
    from types import SimpleNamespace
    def query(args,**kwargs):
        output='123, /usr/bin/python, 2048\n' if '--query-compute-apps=pid,process_name,used_gpu_memory' in args else 'GPU test, 55, 2048, 24000\n'
        return SimpleNamespace(returncode=0,stdout=output)
    monkeypatch.setattr('studio.api.subprocess.run',query)
    with TestClient(create_app(tmp_path)) as c:
        data=c.get('/api/resources').json()
        assert data['compute_processes']==[{'pid':123,'name':'python','used_mb':'2048'}]


@pytest.fixture
def workspace(tmp_path):
    p=tmp_path/'artifacts'/'oracle'/'ame_stakes_1.00'
    write(p/'summary.json',{'args':{'teacher_map':True,'kind':'ame_stakes','seed':801},'successes':0,'total':2,
                            'episodes':[{'success':False,'termination_terms':['base_contact']}]*2})
    write(p/'trace.json',[{'time':.02,'position':[0,0,.3]}])
    p=tmp_path/'artifacts'/'run'
    p.mkdir();(p/'iterations.jsonl').write_text('{"iteration":0,"loss":{"value_function":2}}\n{"iteration":1,"loss":{"value_function":1}}\n{"iteration":')
    write(p/'completed.json',{'last_iteration':1})
    (tmp_path/'README.md').write_text('# Test report\nReal evidence',encoding='utf8')
    return tmp_path


def test_catalog_preserves_teacher_failure_and_handles_partial_logs(workspace):
    c=Catalog(workspace);d=c.get()
    assert len(d['evaluations'])==1 and d['evaluations'][0]['role']=='特权教师'
    assert d['evaluations'][0]['successes']==0 and d['evaluations'][0]['geometry']=='原AME'
    assert c.series(d['runs'][0]['id'])['raw_count']==2
    assert d['reports'][0]['name']=='Test report'


def test_job_operations_never_launch_training_by_default(workspace):
    app=create_app(workspace);client=TestClient(app);headers={'X-Studio-Request':'1'}
    assert client.post('/api/jobs',json={}).status_code==403
    job=client.post('/api/jobs',json={},headers=headers).json()
    assert job['state']=='draft'
    assert client.post('/api/jobs/'+job['id']+'/submit',json={},headers=headers).status_code==403
    assert client.post('/api/jobs/'+job['id']+'/stop',json={},headers=headers).json()['state']=='canceled'
    assert client.post('/api/jobs',json={},headers={**headers,'Origin':'https://untrusted.example'}).status_code==403


def test_api_evidence_and_path_boundary(workspace):
    client=TestClient(create_app(workspace));d=client.get('/api/catalog').json()
    assert client.get('/api/resources').status_code==200
    result=client.get('/api/evaluations/'+d['evaluations'][0]['id']).json()
    assert result['summary']['total']==2 and len(result['trace'])==1
    assert client.get('/api/files/artifacts/oracle/ame_stakes_1.00/summary.json').status_code==200
    assert client.get('/api/files/%2e%2e/%2e%2e/secret.json').status_code==404
    assert client.get('/api/files/.git/config').status_code==404
    assert client.get('/api/catalog').json()==d


def test_config_rejects_unknown_reward_nonfinite_and_invalid_mode(tmp_path):
    for payload in ({'num_envs':511},{'rewards':{'eval_me':3}},{'mode':'student','foothold_aware':True},{'learning_rate':float('nan')},{'initialization':'resume'}):
        with pytest.raises(ValueError):RunConfig.model_validate(payload)
    with pytest.raises(ValueError):command(RunConfig(adapter='nju'),tmp_path,tmp_path/'run',tmp_path/'cfg','python')
    with pytest.raises(ValueError):resolve_checkpoint(tmp_path,'../../secret.pt')


def test_queue_claim_once_and_unknown_blocks(workspace):
    jobs=Jobs(workspace);j=jobs.create(RunConfig().model_dump())
    assert jobs.transition(j['id'],'draft','queued')
    assert not jobs.transition(j['id'],'draft','queued')
    assert jobs.claim()['id']==j['id'];assert jobs.claim() is None
    jobs.update(j['id'],'unknown')
    second=jobs.create(RunConfig().model_dump());jobs.transition(second['id'],'draft','queued')
    assert jobs.claim() is None


def test_exit_zero_is_not_completion(tmp_path):
    with pytest.raises(RuntimeError,match='Missing completed'):checked_training(tmp_path)
    write(tmp_path/'completed.json',{})
    with pytest.raises(RuntimeError,match='without checkpoint'):checked_training(tmp_path)
    with zipfile.ZipFile(tmp_path/'model_0.pt','w') as f:f.writestr('data','checkpoint')
    assert checked_training(tmp_path).name=='model_0.pt'
    write(tmp_path/'failed.json',{'error':'nan'})
    with pytest.raises(RuntimeError,match='failed.json'):checked_training(tmp_path)


def test_override_limits_hash_and_actual_application(tmp_path):
    p=tmp_path/'config.json';write(p,{'ppo':{'learning_rate':.0001},'rewards':{'route_bonus':250}})
    cfg,sha=load(p)
    class Term:weight=500
    class Rewards:route_bonus=Term()
    class Env:rewards=Rewards()
    env=Env();apply_rewards(env,cfg);assert env.rewards.route_bonus.weight==250
    assert len(sha)==64
    write(p,{'ppo':{'schedule':'shell command'}})
    with pytest.raises(ValueError):load(p)


def test_generated_command_is_argument_array_and_overrides_are_wired(tmp_path):
    config=RunConfig(name='safe; never execute',rewards={'route_bonus':250},learning_rate=.0001)
    cmd=command(config,tmp_path,tmp_path/'run',tmp_path/'config.json','/sim/python.sh')
    assert '--studio-config' in cmd and config.name not in cmd
    assert config.overrides()['ppo']['learning_rate']==.0001
    assert '--teacher-map' in cmd and '--navigation' in cmd


def test_legacy_mujoco_schema_is_normalized_without_fabrication():
    raw={'episodes':3,'successes':2,'results':[{'success':True},{'success':False},{'success':True}],'arguments':{'scene':'gap'}}
    normalized=normalize_evaluation(raw)
    assert normalized['total']==3 and len(normalized['episodes'])==3
    assert normalized['args']['scene']=='gap' and raw['episodes']==3


def test_simulation_command_never_trains_and_target_is_explicit(tmp_path):
    cfg=RunConfig(task='simulation',adapter='nju',goal_x=13,goal_y=.6)
    cmd=simulation_command(cfg,tmp_path,tmp_path/'output','/mujoco/python')
    assert cmd[2]=='go2_nav.active_run' and '--video' in cmd
    assert cmd[cmd.index('--goal')+1:cmd.index('--goal')+3]==['13.0','0.6']
    assert 'moe_perceptive.train' not in cmd
    with pytest.raises(ValueError):RunConfig(task='simulation',goal_x=13)


def test_wrong_completed_checkpoint_cannot_pass(tmp_path):
    write(tmp_path/'completed.json',{'last_iteration':199,'additional_iterations':200})
    with zipfile.ZipFile(tmp_path/'model_0.pt','w') as z:z.writestr('data','partial')
    with pytest.raises(RuntimeError,match='missing'):checked_training(tmp_path,200)
    with pytest.raises(RuntimeError,match='budget mismatch'):checked_training(tmp_path,100)


def test_worker_full_flow_with_fake_simulator_only(workspace,monkeypatch):
    from studio.worker import execute
    import studio.worker as worker
    jobs=Jobs(workspace);config=RunConfig(iterations=2,num_envs=8)
    job=jobs.create(config.model_dump())
    class FakeProcess:
        pid=12345
        returncode=0
        def __init__(self,args,**kwargs):
            output=Path(args[args.index('--output')+1]);output.mkdir(parents=True)
            if 'moe_perceptive.train' in args:
                write(output/'completed.json',{'last_iteration':1,'additional_iterations':2})
                with zipfile.ZipFile(output/'model_1.pt','w') as z:z.writestr('data','test fixture only')
            else:
                for i in range(9):write(output/str(i)/'summary.json',{'successes':0,'total':32})
        def poll(self):return 0
    monkeypatch.setattr(worker.subprocess,'Popen',FakeProcess)
    execute(jobs,job,workspace,'/fake/python.sh')
    done=jobs.get(job['id'])
    assert done['state']=='succeeded' and len(done['detail']['evaluation'])==9
    assert len(done['detail']['checkpoint_sha256'])==64


def test_tensorboard_rewards_read_real_event_format(workspace):
    pytest.importorskip('tensorboard')
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto import event_pb2,summary_pb2
    folder=workspace/'artifacts/run';writer=EventFileWriter(str(folder))
    for step,value in enumerate([1.,2.,3.]):
        writer.add_event(event_pb2.Event(step=step,summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag='Train/mean_reward',simple_value=value)])))
    writer.flush();writer.close()
    catalog=Catalog(workspace);run=catalog.get()['runs'][0]
    d=catalog.scalars(run['id'])
    assert d['tag']=='Train/mean_reward' and d['points']==[[0,1.],[1,2.],[2,3.]]
