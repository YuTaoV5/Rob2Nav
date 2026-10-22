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
    manual=config.task=='train' and config.recipe_id in ('navigation_ppo','sideflip_native_ppo','upright_walk_ppo','backflip_native_ppo','frontflip_native_ppo')
    config_path=folder/'studio_config.json';config_path.write_text(json.dumps({'recipe_id':config.recipe_id,'recipe_options':config.recipe_options,'seed':config.seed,'iterations':config.iterations,'num_envs':config.num_envs} if manual else config.overrides(),indent=2))
    (folder/'resolved_config.json').write_text(json.dumps(config.model_dump(),indent=2))
    from .upstreams import pie_python
    runtime=pie_python(root) if config.simulation_policy=='pie' else python if config.simulation_policy=='checkpoint' else mujoco_python
    cmd=simulation_command(config,root,run,runtime) if config.task=='simulation' else command(config,root,run,config_path,python,mujoco_python)
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
                    if stage!='training' or config.recipe_id in ('navigation_ppo','sideflip_native_ppo','upright_walk_ppo','backflip_native_ppo','frontflip_native_ppo'):os.killpg(proc.pid,signal.SIGINT)
                    stopped=True;deadline=time.monotonic()+45
                if stopped and time.monotonic()>deadline:
                    os.killpg(proc.pid,signal.SIGKILL)
                time.sleep(.5)
            if stopped:raise InterruptedError('Stopped at requested boundary; inspect saved checkpoints. Physical state is not restored.')
            if proc.returncode:raise RuntimeError(f'{stage} exited {proc.returncode}')

    if config.task=='simulation':
        launch(cmd,'simulation')
        result=read_json(run/'summary.json')
        total=result.get('total',result.get('episodes')) if result else None
        if total!=1 or not (run/'episode_0.mp4').exists():raise RuntimeError('Simulation output incomplete')
        note='Native checkpoint inference; inspect checkpoint provenance.' if config.simulation_policy=='checkpoint' else 'Historical MoE/IK/NJU skill composition, not new learned perceptive policy.' if config.simulation_policy=='baseline' else 'Reviewed upstream policy inference; not independently validated skill performance.'
        detail.update(result={'successes':result.get('successes'),'total':total},note=note)
        jobs.update(key,'succeeded',detail)
        return
    launch(cmd,'training')
    if config.recipe_id in ('navigation_ppo','sideflip_native_ppo','upright_walk_ppo','backflip_native_ppo','frontflip_native_ppo'):
        if config.recipe_id=='navigation_ppo':
            artifacts=[run/'navigation.zip',run/'manifest.json',run/'navigation_config.json']
        elif config.recipe_id=='backflip_native_ppo' and not config.recipe_options.get('fixstand_roundtrip', False):
            upstream=root.parent/'Rob2Nav-upstreams/GO2_backflip/logs/rsl_rl/go2_backflip'
            matches=list(upstream.glob('*studio_'+key))
            artifacts=sorted(matches[-1].glob('model_*.pt')) if matches else []
        else:
            artifacts=list(run.glob('model_*.pt'))+[run/'completion.json']
        if not artifacts or any(not p.is_file() or p.stat().st_size==0 for p in artifacts):
            raise RuntimeError('Skill trainer exited without required artifacts; inspect training log')
        if config.recipe_id in ('sideflip_native_ppo','upright_walk_ppo','frontflip_native_ppo') or (config.recipe_id=='backflip_native_ppo' and config.recipe_options.get('fixstand_roundtrip', False)):
            completion=read_json(run/'completion.json',{})
            if not completion.get('execution_complete') or completion.get('requested_iterations')!=config.iterations:
                raise RuntimeError('Skill trainer completion manifest does not match the requested budget')
        detail.update(artifacts=[str(p) for p in artifacts],note='Training artifacts exist; independent MuJoCo skill validation is still required.')
        from .recipe_training import skill_evaluation_command
        evaluation=skill_evaluation_command(config,root,run,mujoco_python)
        if evaluation:
            # A rejected skill is a successfully executed experiment, not a
            # qualified deployment. Never infer capability from exit status.
            launch(evaluation,'evaluation')
            result=read_json(run/'evaluation.json',{})
            episodes=result.get('episodes',[])
            start=int(evaluation[evaluation.index('--seed-start')+1])
            if ([e.get('seed') for e in episodes]!=list(range(start,start+4)) or
                result.get('status') not in ('not_accepted','passed_simulation_only') or
                any(type(e.get(k)) is not bool for e in episodes for k in ('passed','motion_pass','constraints_pass'))):
                raise RuntimeError('Incomplete or invalid four-seed skill evaluation')
            accepted=all(e['passed'] and e['motion_pass'] and e['constraints_pass'] for e in episodes)
            if accepted!=(result['status']=='passed_simulation_only'):raise RuntimeError('Evaluation status and episode results disagree')
            detail.update(evaluation=episodes,simulation_acceptance=result['status'],hardware_qualified=False,
                note='Strict four-seed MuJoCo evaluation complete; simulation only. Video review and unified hot-switch validation remain required.')
        elif config.auto_evaluate and config.recipe_id in ('sideflip_native_ppo','backflip_native_ppo','frontflip_native_ppo'):
            detail['note']='Automatic strict acrobatics evaluation requires FIXSTAND roundtrip; this legacy task has not been validated.'
        jobs.update(key,'succeeded',detail)
        return
    ck=checked_training(run,config.iterations)
    detail.update(checkpoint=str(ck.relative_to(root)),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest())
    if config.auto_evaluate:
        output=folder/'evaluation'
        evaluate=[sys.executable,'-m','moe_perceptive.suite','--moe',str(root/'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt'),
                  '--checkpoint',str(ck),'--memory-lifetime',str(config.memory_lifetime),'--sensor-rig',config.sensor_rig,
                  '--video-narrow','--output',str(output)]
        if config.mode=='teacher':evaluate.append('--teacher-map')
        if config.navigation:evaluate.append('--navigation')
        launch(evaluate,'evaluation')
        summaries=list(output.glob('*/summary.json'))
        if len(summaries)!=9:raise RuntimeError('Incomplete strict evaluation: expected nine summaries')
        results=[read_json(p) for p in summaries]
        if any(not d or d.get('total')!=32 for d in results):raise RuntimeError('Invalid first-episode evaluation counts')
        detail['evaluation']=[{'case':p.parent.name,'successes':d['successes'],'total':d['total']} for p,d in zip(summaries,results)]
    detail['note']='Execution completed; capability validation requires inspecting independent evaluation.'
    jobs.update(key,'succeeded',detail)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--python',required=True);parser.add_argument('--mujoco-python',default=os.environ.get('CPU_PY',sys.executable))
    parser.add_argument('--job',help='Internal claimed-job child; parent retains the GPU lock')
    args=parser.parse_args();root=Path(args.root).resolve();jobs=Jobs(root)
    if os.name!='posix':raise SystemExit('Training worker requires a Linux simulator host')
    if args.job:
        job=jobs.get(args.job)
        if job['state']!='preflight':raise RuntimeError('Isolated execution requires an already claimed job')
        try:execute(jobs,job,root,args.python,args.mujoco_python)
        except InterruptedError as e:jobs.update(job['id'],'canceled',{'reason':str(e)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            jobs.update(job['id'],'failed',{'reason':str(e)})
        return
    import fcntl
    with (jobs.directory/'gpu.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        # A dead worker must never cause previously-running jobs to be relaunched.
        for job in jobs.all():
            if job['state'] in ('running','preflight','evaluating'):
                jobs.update(job['id'],'unknown',{'reason':'Worker restarted; inspect process metadata before taking action. Not automatically resumed.'})
        drain_queue(jobs,lambda job:run_isolated(jobs,job,root,args.python,args.mujoco_python))


def run_isolated(jobs,job,root,python,cpu_python):
    """Load current schemas/trainers for EACH job, without restarting a busy parent.

    The parent keeps the single-GPU flock; only its already claimed job is sent
    to the child. Existing training children continue using their original code.
    """
    cmd=[sys.executable,'-m','studio.worker','--root',str(root),'--python',str(python),
         '--mujoco-python',str(cpu_python),'--job',job['id']]
    subprocess.run(cmd,cwd=root,check=True)
    if jobs.get(job['id'])['state'] not in ('succeeded','failed','canceled'):
        raise RuntimeError('Isolated job returned without a terminal execution record')


def drain_queue(jobs,run,clock=time.monotonic,sleep=time.sleep,idle_seconds=60):
    """Only time with no claimed work counts toward worker idle shutdown."""
    idle=clock()
    while clock()-idle<idle_seconds:
        job=jobs.claim()
        if job:
            try:run(job)
            except InterruptedError as e:jobs.update(job['id'],'canceled',{'reason':str(e)})
            except Exception as e:jobs.update(job['id'],'failed',{'reason':str(e)})
            # A training run can take hours; start idle timing AFTER it returns.
            idle=clock()
        else:sleep(1)

if __name__=='__main__':main()
