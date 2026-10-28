"""Reviewed capability and recipe contracts; executable adapters stay allowlisted."""
from urllib.parse import quote
import copy
import json
from pathlib import Path
from .catalog import read_json, clean
from .documents import indexed_manual,indexed_path

REGISTRY=Path(__file__).with_name('skill_registry.json')


def registry():
    data=json.loads(REGISTRY.read_text(encoding='utf-8'))
    from .stability_options import SCHEMAS
    for recipe in data.get('recipes',[]):
        fields=SCHEMAS.get(recipe['id'],{})
        options=recipe.setdefault('defaults',{}).setdefault('recipe_options',{})
        metadata=recipe.setdefault('option_schema',{})
        for key,field in fields.items():
            options.setdefault(key,field['default'])
            metadata[key]={**metadata.get(key,{}),**field}
    return data


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
    if require_ready and recipe['execution'] not in ('native','manual_cli'):raise ValueError(recipe['blocker'])
    if require_ready and recipe.get('defaults',{}).get('kind','mixed')!='mixed' and config.kind!=recipe['defaults']['kind']:
        raise ValueError('Terrain does not match this capability recipe; select the matching recipe or custom experiment')
    return recipe


def capabilities(catalog,jobs):
    data=registry();result=copy.deepcopy(data)
    media_path=Path(__file__).with_name('supermimic_media.json')
    media_overrides=json.loads(media_path.read_text(encoding='utf-8')) if media_path.exists() else {}
    for skill in result['capabilities']:
        # Attach recordings to existing skill IDs without changing recipes or acceptance.
        override=media_overrides.get(skill['id'],{})
        reviewed_media=override.get('curated_media',[])
        if reviewed_media:
            paths={m['path'] for m in reviewed_media}
            skill['curated_media']=copy.deepcopy(reviewed_media)+[m for m in skill.get('curated_media',[]) if m['path'] not in paths]
            skill['preferred_video']=reviewed_media[0]['path']
            for field in ('local_status','description'):
                if field in override:skill[field]=override[field]
        skill['manuals']=[dict(kind=kind,label=label,path=path,url='#reports?path='+quote(path,safe=''))
            for kind,label in [('training','训练手册'),('simulation','仿真手册'),('deployment','部署手册')]
            for path in [indexed_manual(catalog.get('reports',[]),skill['id'],kind)] if any(r['path']==path for r in catalog.get('reports',[]))]
        if skill['id']=='navigation':
            skill['manuals'] += [dict(kind=kind,label=label,path=path,url='#reports?path='+quote(path,safe=''))
                for kind,label,old_path in [('scan_simulation','SCAN仿真手册','skills/navigation/SCAN_SIMULATION.md'),
                                        ('scan_deployment','SCAN实机部署手册','skills/navigation/SCAN_DEPLOYMENT.md'),
                                        ('sensor_calibration','传感器安装与标定','docs/GO2_D435I_MID360_INSTALL_CALIBRATION.md')]
                for path in [indexed_path(catalog.get('reports',[]),old_path)]
                if any(r['path']==path for r in catalog.get('reports',[]))]
        selector=skill.get('evidence',{})
        related=[j for j in jobs if j['config'].get('capability_id')==skill['id']]
        prefixes=selector.get('prefixes',[])+[f"skills/{skill['id']}/evidence/"]+[f"outputs/studio/jobs/{j['id']}/evaluation/" for j in related]
        matches=[e for e in catalog['evaluations'] if any(e['path'].startswith(p) for p in prefixes)
            and (not selector.get('kinds') or e['kind'] in selector['kinds']) and not e.get('diagnostic')]
        skill['evaluations']=matches
        skill['local_videos']=[{'url':'/api/files/'+e['video'],'type':'video','origin':'local','title':e['name'],
            'evaluation_id':e['id'],'label':f"{e['successes']}/{e['total']} · {e['role']} · {e['geometry']}"+(f" · {e['video_scope']}" if e.get('video_scope') else '')}
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
        preferred=skill.get('preferred_video') or next((m['path'] for m in skill.get('curated_media',[])),None)
        if preferred:skill['local_videos'].sort(key=lambda media:not media['url'].endswith(preferred))
        # Completed training is not a proof that the capability passed.
        skill['active_jobs']=sum(j['state'] in ('queued','preflight','running','evaluating') for j in skill['jobs'])
        skill['training_complete']=sum(j['state']=='succeeded' for j in skill['jobs'])
    result['notes']=['作者演示与Rob2Nav独立验证分开。','训练执行完成不等于能力通过。','本体、关节顺序、观测、动作和PD时序不匹配时禁止直接加载权重。']
    return clean(result)
