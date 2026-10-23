"""Build portable job configs and a hash manifest from frozen handoff snapshots."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'artifacts/空翻续训交接_20261023'


def build():
    def config(key):return copy.deepcopy(json.loads((PACK/'原服务器任务快照'/f'{key}.json').read_text(encoding='utf-8'))['config'])
    left='artifacts/空翻续训交接_20261023/左翻父权重/model_12699.pt'
    right='artifacts/空翻续训交接_20261023/右翻恢复快照/model_400.pt'
    configs={}
    c=config('e4baa3665ca04e28')
    c.update(name='迁移_右翻模型400续训600次_20261023',initialization='resume',checkpoint=right,iterations=600,auto_evaluate=True)
    c['recipe_options']['evaluation_seed_start']=27400
    configs['01_右翻恢复600次.json']=c
    c=config('0f602bd56467455c');c['name']='迁移_前翻起跳奖励烟测_20261023'
    configs['02_前翻起跳奖励烟测.json']=copy.deepcopy(c)
    c.update(name='迁移_前翻起跳奖励600次_20261023',num_envs=4096,iterations=600,auto_evaluate=True)
    c['recipe_options']['evaluation_seed_start']=27600
    configs['03_前翻起跳奖励600次.json']=c
    c=config('df310a187da04c32');c.update(name='迁移_关节增益配置烟测_20261023',checkpoint=left)
    configs['04_关节增益配置烟测.json']=copy.deepcopy(c)
    c.update(name='迁移_左翻阻尼1点2续训200次_20261023',num_envs=4096,iterations=200,auto_evaluate=True)
    c['recipe_options'].update(joint_kd=1.2,evaluation_seed_start=27500)
    configs['05_左翻阻尼1点2续训200次.json']=c
    c=config('b24c52e7f3444782');c.update(name='迁移_全程连续性奖励烟测_20261023',checkpoint=left)
    configs['06_全程连续性奖励烟测.json']=c
    c=config('c22023879dd04ee3');c['name']='迁移_自动验收端到端烟测_20261023'
    c['recipe_options']['evaluation_seed_start']=27700
    configs['07_自动验收端到端烟测.json']=c
    dest=PACK/'新服务器任务配置';dest.mkdir(exist_ok=True)
    for name,c in configs.items():
        (dest/name).write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    selected=[p for p in PACK.rglob('*') if p.is_file() and p.name!='资源清单.json']
    for folder in ['side_v12','right_v1']:
        selected.extend((ROOT/'artifacts/web_continuity_iteration_20261023/inputs'/folder).glob('*.npz'))
        selected.extend((ROOT/'artifacts/web_continuity_iteration_20261023/inputs'/folder).glob('*.json'))
    back=ROOT/'artifacts/backflip_v8_2200_verified_20261022'
    selected.extend([back/'model_2200.pt',back/'exported/policy.pt',back/'exported/policy.onnx',back/'exported/export.json',back/'fixstand.json'])
    selected.extend(ROOT/p for p in ['skills/sideflip/train.py','skills/sideflip/continuity.py','skills/sideflip/gains.py',
        'rob2nav/export_actor.py','rob2nav/mujoco_skill.py','rob2nav/target_continuity.py','scripts/evaluate_fixstand_run.py',
        'studio/worker.py','studio/stability_options.py','scripts/acrobatics_handoff.py','scripts/prepare_acrobatics_handoff.py'])
    rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(selected))]
    (PACK/'资源清单.json').write_text(json.dumps(dict(scope='Portable continuation inputs; no capability promotion',files=rows),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{len(configs)} portable configurations; {len(rows)} hashed files')


if __name__=='__main__':build()
