import json
from fastapi.testclient import TestClient
from studio.api import create_app
from studio.config import RunConfig,simulation_command
from studio.jobs import Jobs
from studio.telemetry import checkpoints,replay


def test_checkpoint_command_cannot_silently_use_baseline(tmp_path):
    ck=tmp_path/'artifacts/model_199.pt';ck.parent.mkdir();ck.write_bytes(b'format validated by native runtime')
    cfg=RunConfig(task='simulation',simulation_policy='checkpoint',checkpoint='artifacts/model_199.pt',kind='ame_stakes')
    cmd=simulation_command(cfg,tmp_path,tmp_path/'out','isaac-python')
    assert cmd[:3]==['isaac-python','-m','studio.simulate']
    assert str(ck.resolve()) in cmd and '--max-seconds' in cmd


def test_live_replay_cursor_retries_partial_line_and_job_boundaries(tmp_path):
    jobs=Jobs(tmp_path);j=jobs.create(RunConfig(task='simulation').model_dump())
    folder=tmp_path/'outputs/studio/runs'/j['id'];folder.mkdir(parents=True)
    (folder/'scene.json').write_text(json.dumps({'backend':'MuJoCo','joint_names':['FL_hip_joint']}))
    (folder/'replay.jsonl').write_text('{"time":0.1,"joints":[0.2]}\n{"time":')
    assert replay(folder)['cursor']==1
    with TestClient(create_app(tmp_path)) as c:
        d=c.get(f"/api/jobs/{j['id']}/replay").json()
        assert d['frames'][0]['joints']==[.2]
        assert c.get(f"/api/jobs/{j['id']}/replay?after=1").json()['cursor']==1
        assert c.get('/api/jobs/missing/replay').status_code==404
        assert c.get(f"/api/jobs/{j['id']}/replay?after=-1").status_code==422


def test_details_real_command_nested_loss_no_fake_best(tmp_path):
    jobs=Jobs(tmp_path);j=jobs.create(RunConfig().model_dump());key=j['id']
    folder=tmp_path/'outputs/studio/runs'/key;folder.mkdir(parents=True)
    (folder/'model_9.pt').write_bytes(b'checkpoint')
    (folder/'iterations.jsonl').write_text('{"iteration":9,"loss":{"surrogate":0.3}}\n{"iteration":')
    meta=jobs.folder(key);meta.mkdir(parents=True)
    (meta/'command.json').write_text(json.dumps(['python','-m','moe_perceptive.train','--iterations','10']))
    with TestClient(create_app(tmp_path)) as c:
        d=c.get(f'/api/jobs/{key}/telemetry').json()
        assert d['raw_count']==1 and d['rows'][0]['loss/surrogate']==.3
        assert '--iterations 10' in d['command']
        assert d['weights']['selected']['iteration']==9
        assert not d['weights']['validated_best']
        assert c.get('/api/jobs/missing/telemetry').status_code==404


def test_best_reward_only_scores_existing_exact_saved_steps(tmp_path,monkeypatch):
    jobs=Jobs(tmp_path);j=jobs.create(RunConfig().model_dump())
    folder=tmp_path/'outputs/studio/runs'/j['id'];folder.mkdir(parents=True)
    for n in (9,19,29):(folder/f'model_{n}.pt').write_bytes(b'weight')
    monkeypatch.setattr('studio.telemetry.metrics',lambda *a,**k:({'series':{'Train/mean_reward':[[9,10],[15,999],[19,20],[29,5]]}},''))
    d=checkpoints(tmp_path,j)
    assert d['selected']['iteration']==19 and d['selected']['reward']==20
    assert not d['validated_best']


def test_tensorboard_real_event_tags_and_saved_checkpoint_ranking(tmp_path):
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from studio.telemetry import telemetry
    jobs=Jobs(tmp_path);j=jobs.create(RunConfig().model_dump())
    folder=tmp_path/'outputs/studio/runs'/j['id'];folder.mkdir(parents=True)
    writer=EventFileWriter(str(folder))
    for step,reward in ((9,10),(19,30),(29,5)):
        writer.add_event(Event(wall_time=1.,step=step,summary=Summary(value=[
            Summary.Value(tag='Train/mean_reward',simple_value=reward),
            Summary.Value(tag='Loss/surrogate',simple_value=1./step)])))
        (folder/f'model_{step}.pt').write_bytes(b'checkpoint')
    writer.close()
    data=telemetry(tmp_path,jobs,j['id'],['Loss/surrogate'])
    assert set(data['scalars']['tags'])=={'Train/mean_reward','Loss/surrogate'}
    assert len(data['scalars']['series']['Loss/surrogate'])==3
    assert data['weights']['selected']['iteration']==19
