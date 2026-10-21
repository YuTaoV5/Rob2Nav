"""Explicit inference-only launch from a known queue job; shares the GPU lock."""
import os
from pathlib import Path
import subprocess
import sys
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from .config import RunConfig, simulation_command
from .jobs import Jobs
from .telemetry import checkpoints

class CheckpointLaunch(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    job_id:str=Field(min_length=1,max_length=32,pattern=r'^[a-zA-Z0-9_-]+$')
    checkpoint:str|None=None
    kind:Literal['flat','stairs','platform','ame_stakes','ame_alternate']='flat'
    difficulty:float=Field(default=.5,ge=0,le=1)
    max_seconds:float=Field(default=15,ge=1,le=120)
    seed:int=Field(default=801,ge=0,le=2147483647)

def resolve_launch(root,jobs,a):
    try:source=jobs.get(a.job_id)
    except KeyError:raise ValueError('训练任务不存在')
    cfg=source['config']
    if cfg.get('task')=='simulation' or cfg.get('robot_id','go2')!='go2' or cfg.get('adapter','moe')!='moe' or cfg.get('algorithm_id','moe_terrain_ppo')!='moe_terrain_ppo':
        raise ValueError('该任务尚无匹配的原生实时推理适配器；禁止替换为其他模型')
    weights=checkpoints(root,source);selected=weights.get('selected')
    path=a.checkpoint or (selected or {}).get('path')
    if not path or path not in {c['path'] for c in weights['candidates']}:raise ValueError('只能选择此训练任务中已保存的检查点')
    # Observation/sensor contract remains the training job's contract. Checkpoint
    # provenance is checked again by studio.simulate in the simulator interpreter.
    values=dict(cfg)
    values.update(name=('实时权重仿真 · '+source['name'])[:80],task='simulation',simulation_policy='checkpoint',
                  checkpoint=path,initialization='base',kind=a.kind,difficulty=a.difficulty,
                  max_seconds=a.max_seconds,seed=a.seed,body_height_target_m=None,swing_clearance_target_m=None)
    return source,RunConfig.model_validate(values),weights

def mount_checkpoint_live(app,root):
    root=Path(root).resolve()
    from .catalog import Catalog
    assets=Catalog(root)
    @app.get('/api/model-assets/{key}/download')
    def download_model(key:str):
        from .model_registry import asset_path
        m=next((m for m in assets.get()['models'] if m['id']==key),None)
        if m is None:raise HTTPException(404,'Model not indexed')
        try:p=asset_path(root,m['path'])
        except ValueError as e:raise HTTPException(404,str(e))
        return FileResponse(p,filename=p.name,media_type='application/octet-stream')

    @app.post('/api/checkpoint-simulation')
    def start_checkpoint(a:CheckpointLaunch):
        jobs=Jobs(root)
        try:source,config,weights=resolve_launch(root,jobs,a)
        except ValueError as e:raise HTTPException(422,str(e))
        runtime=Path(os.environ.get('ROB2NAV_ISAAC_PYTHON','/isaac-sim/python.sh'))
        if os.name!='posix' or not runtime.is_file():raise HTTPException(409,'当前主机未安装匹配的 Isaac 推理运行时')
        # No queue worker is started here: this process can execute only the
        # freshly created simulation job, even while training submission is off.
        import fcntl
        lock=(jobs.directory/'gpu.lock').open('a')
        try:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise HTTPException(409,'GPU正在执行任务，请等待当前任务结束')
            if any(j['state'] in ('queued','preflight','running','evaluating','unknown') for j in jobs.all()):
                raise HTTPException(409,'队列有活动或待核对任务，先完成或核对其状态')
            simulation_command(config,root,root/'outputs/studio/probe',runtime)
            job=jobs.create(config.model_dump());folder=jobs.folder(job['id']);folder.mkdir(parents=True,exist_ok=True)
            detail=dict(stage='simulation',source_job=source['id'],selection_basis=weights['criterion'],checkpoint=config.checkpoint,
                        note='仅执行指定检查点推理；训练提交保持原设置',hardware_enabled=False)
            jobs.update(job['id'],'preflight',detail)
            with (folder/'launcher.log').open('wb') as log:
                try:
                    subprocess.Popen([sys.executable,'-m','studio.checkpoint_runner','--root',str(root),'--job',job['id'],
                                      '--python',str(runtime),'--lock-fd',str(lock.fileno())],
                                     cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
                except OSError as e:
                    jobs.update(job['id'],'failed',dict(reason=str(e)));raise HTTPException(500,'推理进程启动失败')
            return jobs.get(job['id'])
        finally:lock.close()
