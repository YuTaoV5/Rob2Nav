"""Reviewed capability and recipe contracts; executable adapters stay allowlisted."""
import copy
import json
from pathlib import Path
from .catalog import read_json, clean

REGISTRY=Path(__file__).with_name('skill_registry.json')


def registry():
    return json.loads(REGISTRY.read_text(encoding='utf-8'))


def resolve_recipe(config,require_ready=False):
    if not config.recipe_id:
        if config.robot_id!='go2' or config.algorithm_id!='moe_terrain_ppo':
            raise ValueError('Legacy adapter requires the Go2 / MoE terrain PPO contract')
        return None
    data=registry();recipe=next((r for r in data['recipes'] if r['id']==config.recipe_id),None)
    if recipe is None:raise ValueError('Unknown reviewed training recipe')
    if config.robot_id!=recipe['robot'] or config.algorithm_id!=recipe['algorithm'] or config.adapter!=recipe['adapter']:
        raise ValueError('Robot / algorithm / adapter does not match the selected recipe; weights are not interchangeable')
    if config.capability_id and config.capability_id not in recipe['capabilities']:
        raise ValueError('Capability is not supported by this recipe')
    if require_ready and recipe['execution']!='native':raise ValueError(recipe['blocker'])
    if require_ready and recipe.get('defaults',{}).get('kind','mixed')!='mixed' and config.kind!=recipe['defaults']['kind']:
        raise ValueError('Terrain does not match this capability recipe; select the matching recipe or custom experiment')
    return recipe


def capabilities(catalog,jobs):
    data=registry();result=copy.deepcopy(data)
    for skill in result['capabilities']:
        selector=skill.get('evidence',{})
        related=[j for j in jobs if j['config'].get('capability_id')==skill['id']]
        prefixes=selector.get('prefixes',[])+[f"outputs/studio/jobs/{j['id']}/evaluation/" for j in related]
        matches=[e for e in catalog['evaluations'] if any(e['path'].startswith(p) for p in prefixes)
            and (not selector.get('kinds') or e['kind'] in selector['kinds']) and not e.get('diagnostic')]
        skill['evaluations']=matches
        skill['local_videos']=[{'url':'/api/files/'+e['video'],'type':'video','origin':'local','title':e['name'],
            'evaluation_id':e['id'],'label':f"{e['successes']}/{e['total']} · {e['role']} · {e['geometry']}"}
            for e in matches if e.get('video')]
        for job in related:
            for video in catalog['videos']:
                if video['path']==f"outputs/studio/runs/{job['id']}/episode_0.mp4" and not any(m['url'].endswith(video['path']) for m in skill['local_videos']):
                    skill['local_videos'].append(dict(url='/api/files/'+video['path'],type='video',origin='local',title=job['name'],label='本地策略推理回放，能力验收另见独立评估'))
        skill['jobs']=related
        available={v['path'] for v in catalog['videos']}
        for media in skill.get('curated_media',[]):
            if media['path'] in available and not any(m['url'].endswith(media['path']) for m in skill['local_videos']):
                skill['local_videos'].append(dict(url='/api/files/'+media['path'],type='video',origin='local',title=media['title'],label=media['label']))
        # Completed training is not a proof that the capability passed.
        skill['active_jobs']=sum(j['state'] in ('queued','preflight','running','evaluating') for j in skill['jobs'])
        skill['training_complete']=sum(j['state']=='succeeded' for j in skill['jobs'])
    result['notes']=['作者演示与Rob2Nav独立验证分开。','训练执行完成不等于能力通过。','本体、关节顺序、观测、动作和PD时序不匹配时禁止直接加载权重。']
    return clean(result)
