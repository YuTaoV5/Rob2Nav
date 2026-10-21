"""One bounded, CPU-only live skill process. No arbitrary paths or robot transport."""
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel,ConfigDict,Field
from typing import Literal


class Launch(BaseModel):
    model_config=ConfigDict(extra='forbid')
    skill:Literal['stairs','platform','multiterrain','backflip','sideflip','upright_walk']


class CompositionStep(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    skill:str=Field(min_length=1,max_length=64)
    seconds:float=Field(default=2.5,gt=0,le=120)
    velocity:tuple[float,float,float]=(0.,0.,0.)
    trigger:bool=False


class Composition(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    steps:list[CompositionStep]=Field(min_length=2,max_length=8)
    cycles:int=Field(default=1,ge=1,le=4)
    research_stunts:bool=False


class Camera(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    azimuth:float=Field(default=130,ge=-720,le=720)
    elevation:float=Field(default=-25,ge=-89,le=-5)
    distance:float=Field(default=3,ge=.5,le=18)
    pan_x:float=Field(default=0,ge=-10,le=10)
    pan_y:float=Field(default=0,ge=-10,le=10)


def mount_live_skills(app,root,python):
    from .checkpoint_live import mount_checkpoint_live
    mount_checkpoint_live(app,root)
    root=Path(root);lock=threading.RLock();current={};proc=None;last_client=0.
    source=Path('/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip')
    def resource(relative,fallback):
        packaged=root/'artifacts/release_20261021/weights'/relative
        return packaged if packaged.is_file() else fallback
    def read(p,default):
        try:return json.loads(p.read_text())
        except (OSError,ValueError):return default
    def stop():
        nonlocal proc
        if proc is not None and proc.poll() is None:
            import os
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=3)
        if current:current['phase']='STOPPED'
    closed=threading.Event()
    @app.on_event("shutdown")
    def shutdown():
        closed.set()
        with lock:stop()
    def watchdog():
        while not closed.wait(1):
            with lock:
                if proc is not None and proc.poll() is None and time.monotonic()-last_client>8:stop()
    threading.Thread(target=watchdog,daemon=True).start()

    @app.post('/api/live-skills/start')
    def start(a:Launch):
        nonlocal proc,last_client
        with lock:
            stop();key=uuid.uuid4().hex[:12];out=root/'outputs/live_skills'/key
            stream=out/'stream';stream.mkdir(parents=True)
            note='';policy=None
            if a.skill in ('stairs','platform','multiterrain'):
                module='go2_nav.terrain_run'
                args=['--scene',{'multiterrain':'mixed'}.get(a.skill,a.skill),'--locomotion','moe','--max-seconds','45','--episodes','1','--seed','801','--no-jump','--output',str(out/'evaluation')]
                note='RoboGauge预训练MoE + 感知地形控制；不是Isaac感知微调检查点的迁移验收。'
            elif a.skill=='backflip':
                module='skills.backflip.simulate'
                policy=resource('backflip/selected/exported/policy.onnx',source/'logs/rsl_rl/go2_backflip/2026-10-21_00-04-59_rob2nav_reproduction/exported/policy.onnx')
                args=['--source',str(source),'--onnx',str(policy),'--physics-hz','1000','--seconds','8','--output',str(out/'evaluation')]
                note='本地5000轮策略；严格落地和电机约束未通过。'
            else:
                module='rob2nav.mujoco_skill'
                policy=root/('outputs/sideflip_dc_adapt_20261021/probe_5200/policy.pt' if a.skill=='sideflip' else 'outputs/upright_walk_full_20261021/exported_final/policy.pt')
                policy=resource('sideflip/selected/exported/policy.pt' if a.skill=='sideflip' else 'upright_walk/representative_failed_baseline/exported/policy.pt',policy)
                args=['--kind',a.skill,'--policy',str(policy),'--backflip-source',str(source),'--physics-hz','1000','--seconds','10','--output',str(out/'evaluation')]
                if a.skill=='sideflip':args+=['--motion','/root/gpufree-data/capability-work/sideflip/reference.npz']
                note='侧翻动作已验证8/8，额定速度未通过。' if a.skill=='sideflip' else '实验策略：MuJoCo起立/行走未通过，展示实际失败行为。'
            if policy is not None and not policy.is_file():raise HTTPException(409,'该技能导出模型尚未就绪')
            log=(out/'process.log').open('wb')
            try:proc=subprocess.Popen([str(python),'-m','studio.live_runner','--stream',str(stream),'--module',module,'--',*args],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            finally:log.close()
            current.clear();current.update(id=key,skill=a.skill,phase='STARTING',note=note,stream=str(stream),hardware_enabled=False)
            last_client=time.monotonic()
            return {k:v for k,v in current.items() if k!='stream'}

    @app.post('/api/live-skills/composition/start')
    def start_composition(a:Composition):
        nonlocal proc,last_client
        from rob2nav.deployment.sequence import Step
        profiles={p.stem:read(p,{}) for p in (root/'deployment/skills').glob('*.json')}
        if a.steps[0].skill!='stand' or a.steps[-1].skill!='stand':raise HTTPException(422,'编排必须从 stand 开始并以 stand 结束')
        if a.cycles>1 and any(s.skill=='sideflip69' for s in a.steps) and not a.research_stunts:
            raise HTTPException(422,'侧翻重复演示需要显式开启研究仿真')
        allowed={'stand','stand_soft','stand_stiff','walk_mlp','walk_moe'}|({'sideflip69'} if a.research_stunts else set())
        steps=[]
        for cycle in range(a.cycles):
            steps.extend(s.model_dump() for s in a.steps if cycle==0 or s is not a.steps[0])
        if len(steps)>32:raise HTTPException(422,'最多32个展开步骤')
        for raw in steps:
            if raw['skill'] not in allowed or raw['skill'] not in profiles:raise HTTPException(422,'技能未接入统一MuJoCo编排或仍在隔离中')
            if raw['trigger']!=bool(profiles[raw['skill']].get('atomic')):raise HTTPException(422,'原子技能需要显式触发，普通技能不能触发')
            try:Step(**raw).checked()
            except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
        budget=min(1800.,max(30.,sum(s['seconds'] for s in steps)+len(steps)*8+15))
        with lock:
            stop();key=uuid.uuid4().hex[:12];out=root/'outputs/live_skills'/key
            stream=out/'stream';stream.mkdir(parents=True)
            sequence=out/'sequence.json';sequence.write_text(json.dumps({'steps':steps},ensure_ascii=False))
            args=['--root',str(root),'--output',str(out/'evaluation'),'--asset','source','--sequence',str(sequence),'--max-seconds',str(budget),'--seed','9315']
            if a.research_stunts:args.append('--atomic-cycle')
            log=(out/'process.log').open('wb')
            try:proc=subprocess.Popen([str(python),'-m','studio.live_runner','--stream',str(stream),'--module','rob2nav.deployment.verify_mujoco','--',*args],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            finally:log.close()
            current.clear();current.update(id=key,skill='composition',phase='STARTING',note='统一控制器真实MuJoCo编排；侧翻仅研究仿真，所有配置均未获实机资格。',stream=str(stream),hardware_enabled=False,steps=len(steps),cycles=a.cycles)
            last_client=time.monotonic()
            return {k:v for k,v in current.items() if k!='stream'}

    @app.get('/api/live-skills/state')
    def state():
        nonlocal last_client
        with lock:
            last_client=time.monotonic()
            if not current:return {'phase':'IDLE'}
            s={k:v for k,v in current.items() if k!='stream'}
            s.update(read(Path(current['stream'])/'state.json',{}))
            result=read(Path(current['stream']).parent/'evaluation/summary.json',{})
            if result:
                s['result']={k:result[k] for k in ['successes','total','hardware_constraints_pass','metrics','collisions','falls'] if k in result}
                if current.get('skill')=='composition':s['result'].update({k:result[k] for k in ['passed','reason','sequence','max_torque_step_Nm','maximum_joint_speed_ratio','commits'] if k in result})
                if 'total' not in s['result'] and isinstance(result.get('episodes'),int):s['result']['total']=result['episodes']
            if proc is not None and proc.poll() is not None:
                s['exit_code']=proc.returncode
                s['phase']='STOPPED' if current['phase']=='STOPPED' else 'COMPLETED' if proc.returncode==0 else 'FAILED'
                if s['phase']=='FAILED':s['error']=(Path(current['stream']).parent/'process.log').read_text(errors='replace')[-2400:]
            return s

    @app.get('/api/live-skills/frame.jpg')
    def frame():
        with lock:
            p=Path(current.get('stream','/nonexistent'))/'frame.jpg'
            if not p.is_file():raise HTTPException(404,'等待第一帧')
            return Response(p.read_bytes(),media_type='image/jpeg',headers={'Cache-Control':'no-store'})

    @app.post('/api/live-skills/camera')
    def camera(a:Camera):
        with lock:
            if not current:raise HTTPException(409,'尚未启动仿真')
            p=Path(current['stream'])/'camera.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(a.model_dump()));tmp.replace(p)
            return {'accepted':True}

    @app.post('/api/live-skills/stop')
    def halt():
        with lock:stop()
        return {'stopped':True}
