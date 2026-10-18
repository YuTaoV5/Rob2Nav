"""Single-GPU worker; launch only after explicit API submission on Linux."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
from .jobs import Jobs
from .config import RunConfig,command,simulation_command
from .catalog import read_json


def checked_training(folder,expected_iterations=None):
    if (folder/'failed.json').exists():raise RuntimeError('Training wrote failed.json: '+str(read_json(folder/'failed.json')))
    if not (folder/'completed.json').exists():raise RuntimeError('Missing completed.json even if launcher returned zero')
    completed=read_json(folder/'completed.json',{})
    if expected_iterations is not None and completed.get('additional_iterations')!=expected_iterations:
        raise RuntimeError('Completion iteration budget mismatch')
    checkpoints=sorted(folder.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    if not checkpoints:raise RuntimeError('Training completed without checkpoint')
    checkpoint=checkpoints[-1]
    if completed.get('last_iteration') is not None and checkpoint.name!=f"model_{completed['last_iteration']}.pt":
        raise RuntimeError('Final completed checkpoint is missing')
    with zipfile.ZipFile(checkpoint) as z:
        if z.testzip():raise RuntimeError('Checkpoint failed archive integrity')
    return checkpoint


def execute(jobs,job,root,python,mujoco_python=None):
    key=job['id'];folder=jobs.folder(key);folder.mkdir(parents=True,exist_ok=True)
    config=RunConfig.model_validate(job['config']);run=root/'outputs/studio/runs'/key
    config_path=folder/'studio_config.json';config_path.write_text(json.dumps(config.overrides(),indent=2))
    (folder/'resolved_config.json').write_text(json.dumps(config.model_dump(),indent=2))
    cmd=simulation_command(config,root,run,mujoco_python) if config.task=='simulation' else command(config,root,run,config_path,python)
    (folder/'command.json').write_text(json.dumps(cmd,indent=2))
    detail={'stage':'training','run':str(run.relative_to(root)),
            'config_sha256':hashlib.sha256((folder/'resolved_config.json').read_bytes()).hexdigest(),
            'override_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest()}

    def launch(args,stage):
        if (folder/'cancel.flag').exists():raise InterruptedError('Canceled before stage')
        with (folder/(stage+'.log')).open('w') as log:
            proc=subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            detail.update(pid=proc.pid,stage=stage)
            (folder/'process.json').write_text(json.dumps({'pid':proc.pid,'started':time.time(),'command':args}))
            jobs.update(key,'running' if stage=='training' else 'evaluating',detail)
            stopped=False;deadline=None
            while proc.poll() is None:
                (jobs.directory/'worker_heartbeat.json').write_text(json.dumps({'time':time.time(),'job':key,'pid':os.getpid()}))
                if (folder/'cancel.flag').exists() and not stopped:
                    # MoE checks the same flag at the next rollout boundary,
                    # saves an atomic stop checkpoint, then exits. Other jobs
                    # have no cooperative checkpoint contract.
                    if stage!='training':os.killpg(proc.pid,signal.SIGINT)
                    stopped=True;deadline=time.monotonic()+45
                if stopped and time.monotonic()>deadline:
                    os.killpg(proc.pid,signal.SIGKILL)
                time.sleep(.5)
            if stopped:raise InterruptedError('Stopped at requested boundary; inspect saved checkpoints. Physical state is not restored.')
            if proc.returncode:raise RuntimeError(f'{stage} exited {proc.returncode}')

    if config.task=='simulation':
        launch(cmd,'simulation')
        result=read_json(run/'summary.json')
        if not result or result.get('episodes')!=1 or not (run/'episode_0.mp4').exists():raise RuntimeError('Simulation output incomplete')
        detail.update(result={'successes':result.get('successes'),'total':result['episodes']},note='Historical MoE/IK/NJU skill composition, not new learned perceptive policy.')
        jobs.update(key,'succeeded',detail)
        return
    launch(cmd,'training')
    ck=checked_training(run,config.iterations)
    detail.update(checkpoint=str(ck.relative_to(root)),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest())
    if config.auto_evaluate:
        output=folder/'evaluation'
        evaluate=[sys.executable,'-m','moe_perceptive.suite','--moe',str(root/'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt'),
                  '--checkpoint',str(ck),'--navigation','--memory-lifetime',str(config.memory_lifetime),'--sensor-rig',config.sensor_rig,
                  '--video-narrow','--output',str(output)]
        if config.mode=='teacher':evaluate.append('--teacher-map')
        launch(evaluate,'evaluation')
        summaries=list(output.glob('*/summary.json'))
        if len(summaries)!=9:raise RuntimeError('Incomplete strict evaluation: expected nine summaries')
        results=[read_json(p) for p in summaries]
        if any(not d or d.get('total')!=32 for d in results):raise RuntimeError('Invalid first-episode evaluation counts')
        detail['evaluation']=[{'case':p.parent.name,'successes':d['successes'],'total':d['total']} for p,d in zip(summaries,results)]
    detail['note']='Execution completed; capability validation requires inspecting independent evaluation.'
    jobs.update(key,'succeeded',detail)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--python',required=True);parser.add_argument('--mujoco-python',default='/root/gpufree-data/envs/go2-mapless/bin/python')
    args=parser.parse_args();root=Path(args.root).resolve();jobs=Jobs(root)
    if os.name!='posix':raise SystemExit('Training worker requires a Linux simulator host')
    import fcntl
    with (jobs.directory/'gpu.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        # A dead worker must never cause previously-running jobs to be relaunched.
        for job in jobs.all():
            if job['state'] in ('running','preflight','evaluating'):
                jobs.update(job['id'],'unknown',{'reason':'Worker restarted; inspect process metadata before taking action. Not automatically resumed.'})
        idle=time.monotonic()
        while time.monotonic()-idle<60:
            job=jobs.claim()
            if job:
                idle=time.monotonic()
                try:execute(jobs,job,root,args.python,args.mujoco_python)
                except InterruptedError as e:jobs.update(job['id'],'canceled',{'reason':str(e)})
                except Exception as e:jobs.update(job['id'],'failed',{'reason':str(e)})
            else:time.sleep(1)

if __name__=='__main__':main()
