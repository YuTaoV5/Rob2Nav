"""Isolated real-checkpoint integration test. Does not create training in the live queue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from fastapi.testclient import TestClient
from studio.api import create_app
from studio.config import RunConfig
from studio.jobs import Jobs

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    import fcntl
    global_lock=(root/'outputs/studio/gpu.lock').open('a')
    fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if any(j['state'] in ('queued','preflight','running','evaluating','unknown') for j in Jobs(root).all()):
        raise RuntimeError('Live queue is busy or needs reconciliation; no validation process launched')
    fixture=out/'workspace';fixture.mkdir()
    for child in root.iterdir():
        if child.is_dir() and not child.name.startswith('.') and child.name not in ('outputs','artifacts'):
            (fixture/child.name).symlink_to(child,target_is_directory=True)
    jobs=Jobs(fixture)
    config=RunConfig(name='验证夹具：既有阶梯199检查点（未训练）',navigation=False,mode='student',arrival_aware=False,
                     kind='stairs',course='original',memory_lifetime=4,sensor_rig='original',capability_id='stairs').model_dump()
    source=jobs.create(config);jobs.update(source['id'],'snapshot',{'note':'Imported existing checkpoint for isolated inference validation; no training performed'})
    folder=fixture/'outputs/studio/runs'/source['id'];folder.mkdir(parents=True)
    checkpoint=root/'artifacts/release_20261021/weights/stairs/manual_reproduction/model_199.pt'
    os.link(checkpoint,folder/checkpoint.name)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();headers={'X-Studio-Request':'1'}
    report={'source_checkpoint':str(checkpoint.relative_to(root)),'sha256':digest,'scope':'6s native physics integration smoke; no training and no capability success claim'}
    with TestClient(create_app(fixture,allow_training=False)) as c:
        assert c.get('/api/resources').json()['training_enabled'] is False
        import fcntl
        with (jobs.directory/'gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert c.post('/api/checkpoint-simulation',headers=headers,json={'job_id':source['id']}).status_code==409
        response=c.post('/api/checkpoint-simulation',headers=headers,json={'job_id':source['id'],'kind':'stairs','max_seconds':6,'difficulty':.5})
        assert response.status_code==200,response.text
        key=response.json()['id'];started=time.monotonic();counts=[];observed_running=False
        for _ in range(300):
            j=jobs.get(key);d=c.get('/api/jobs/'+key+'/replay').json()
            if d['frames']:
                counts.append(len(d['frames']));observed_running |= j['state'] in ('preflight','running','evaluating')
            if j['state'] in ('succeeded','failed','canceled'):break
            time.sleep(1)
        assert j['state']=='succeeded',j
        run=fixture/'outputs/studio/runs'/key
        summary=json.loads((run/'summary.json').read_text());scene=json.loads((run/'scene.json').read_text())
        assert summary['checkpoint_sha256']==digest
        assert len(set(counts))>1 and observed_running,counts
        assert scene['backend']=='Isaac Sim' and (run/'episode_0.mp4').stat().st_size>1000
        report.update(simulation_job=key,source_job=source['id'],final_state=j['state'],frames=len(d['frames']),
                      observed_live_counts=counts,wall_seconds=time.monotonic()-started,backend=scene['backend'],
                      final_sim_time=d['frames'][-1]['time'],training_enabled=False,gpu_lock_conflict_rejected=True,
                      exact_checkpoint_hash_verified=True,video_bytes=(run/'episode_0.mp4').stat().st_size)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
