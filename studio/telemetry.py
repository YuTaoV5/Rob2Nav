"""Bounded, read-only telemetry for a known Studio job. Never unpickle weights."""
import json
import math
import re
import shlex
from pathlib import Path
from .catalog import read_json, clean
_cache={}


def rows(path, limit=2000):
    result=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
            try:
                value=json.loads(line)
                if isinstance(value,dict):result.append(value)
            except ValueError:pass
    stride=max(1,math.ceil(len(result)/max(1,limit-1)))
    sample=result[::stride]
    if result and (not sample or sample[-1] is not result[-1]):sample.append(result[-1])
    return sample,len(result)


def metrics(folder, tags=None, exact=False):
    events=list(folder.glob('events.out.tfevents.*'))
    if not events:return {},'没有TensorBoard事件；显示已记录的轮次指标。'
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        fingerprint=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(events))
        cached=_cache.get(str(folder))
        if cached and cached[0]==fingerprint:acc=cached[1]
        else:
            acc=EventAccumulator(str(folder),size_guidance={'scalars':0});acc.Reload()
            if len(_cache)>32:_cache.pop(next(iter(_cache)))
            _cache[str(folder)]=(fingerprint,acc)
        available=acc.Tags().get('scalars',[])
        selected=available if tags is None else [t for t in tags if t in available]
        result={}
        for tag in selected[:16]:
            values=acc.Scalars(tag);stride=1 if exact else max(1,math.ceil(len(values)/1999));sample=values[::stride]
            if values and (not sample or sample[-1] is not values[-1]):sample.append(values[-1])
            result[tag]=[[v.step,v.value] for v in sample]
        return {'tags':available,'series':result},''
    except ImportError:return {},'需要安装tensorboard才能读取事件。'


def checkpoints(root,job):
    folder=root/'outputs/studio/runs'/job['id']
    candidates=[]
    for p in folder.glob('model_*.pt'):
        match=re.fullmatch(r'model_(\d+)\.pt',p.name)
        if match:candidates.append({'path':p.relative_to(root).as_posix(),'iteration':int(match[1]),'bytes':p.stat().st_size})
    candidates.sort(key=lambda c:c['iteration'])
    # All current Studio strict evaluations refer to the final checkpoint only.
    # Do not claim an unmeasured earlier checkpoint is validation-best.
    final=job.get('detail',{}).get('checkpoint')
    selected=next((c for c in candidates if c['path']==final),None) or (candidates[-1] if candidates else None)
    scalar,_=metrics(folder,['Train/mean_reward'],exact=True) if candidates else ({},'')
    scores=dict(scalar.get('series',{}).get('Train/mean_reward',[]))
    scored=[dict(c,reward=scores[c['iteration']]) for c in candidates if c['iteration'] in scores and math.isfinite(scores[c['iteration']])]
    if scored:selected=max(scored,key=lambda c:c['reward'])
    source=job.get('config',{}).get('checkpoint') if job.get('config',{}).get('task')=='simulation' else None
    if not selected and source:
        from .config import resolve_checkpoint
        try:
            p=resolve_checkpoint(root,source)
            selected={'path':source,'iteration':None,'bytes':p.stat().st_size}
        except ValueError:pass
    return {'candidates':candidates,'selected':selected,'criterion':'本次仿真输入权重' if source else '保存点训练平均奖励最高' if scored else '最终评估权重' if final and selected else '最新已保存权重',
            'validated_best':False,'note':'尚无同协议多检查点排名；不将最新权重称为已验证最佳。'}


def telemetry(root,jobs,key,tags=None):
    job=jobs.get(key);folder=root/'outputs/studio/runs'/key;meta=jobs.folder(key)
    series,count=rows(folder/'iterations.jsonl');scalar,note=metrics(folder,tags)
    argv=read_json(meta/'command.json',[])
    for row in series:
        for name,value in row.get('loss',{}).items():row['loss/'+name]=value
    return clean({'job':job,'status':read_json(folder/'status.json',{}),'rows':series,'raw_count':count,
        'scalars':scalar,'note':note,'weights':checkpoints(root,job),'argv':argv,
        'command':shlex.join(argv) if argv else None,'resolved':read_json(meta/'resolved_config.json',job['config']),
        'skill_contract':{'robot':job['config'].get('robot_id','go2'),'algorithm':job['config'].get('algorithm_id','moe_terrain_ppo'),
            'capability':job['config'].get('capability_id'),'recipe':job['config'].get('recipe_id'),
            'io':read_json(folder/'io.json',{}),'deployment_ready':False,
            'note':'统一清单不是实机部署许可；仍需归一化、关节/动作/PD合约及迁移验证。'}})


def replay(folder, after=0):
    # A bounded stream: cursor is a JSONL line number, incomplete lines are retried.
    p=folder/'replay.jsonl';frames=[];cursor=0
    if p.exists():
        with p.open(encoding='utf-8') as stream:
            for index,line in enumerate(stream):
                if index<after:cursor=index+1;continue
                try:frame=json.loads(line)
                except ValueError:break
                frames.append(frame);cursor=index+1
                if len(frames)>=600:break
    return clean({'scene':read_json(folder/'scene.json',{}),'frames':frames,'cursor':cursor})
