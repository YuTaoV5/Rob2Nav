"""Exact, allowlisted intent grammar; ASR/LLM text never becomes code."""
import re
from .places import normalized

SKILLS = {
    'backflip': dict(name='后空翻', profile='backflip60', aliases=['后空翻', '后翻', '做个后空翻', '来个后空翻']),
    'sideflip': dict(name='侧空翻', profile='sideflip69', aliases=['侧空翻', '侧翻', '做个侧空翻', '来个侧空翻']),
    'handstand': dict(name='倒立', profile='upright48', aliases=['倒立', '倒立走', '做个倒立', '来个倒立']),
}


def parse(text):
    if not isinstance(text, str) or not 0 < len(text) <= 200: return {'kind': 'unknown'}
    text = re.sub(r'<\|[^>]*\|>', '', text)
    key = normalized(text)
    for wake in ('小狗', '机器狗', '狗狗'):
        if key.startswith(wake): key = key[len(wake):]; break
    key = re.sub(r'^(请你|请|帮我)', '', key)
    key = re.sub(r'(一下|吧)$', '', key)
    if key in ('急停', '紧急停止', '立刻急停', 'emergencystop'): return {'kind': 'emergency'}
    if key in ('停', '停止', '停下', '暂停', '取消导航', '取消任务', '停止动作', '别动', '站好', '站立', 'stop'):
        return {'kind': 'stop'}
    if key in ('开始控制', '打开控制', '启用控制', '启用', 'enable'): return {'kind': 'enable'}
    if key in ('关闭控制', '禁用控制', '禁用', 'disable'): return {'kind': 'disable'}
    if key in ('状态', '报告状态', '现在什么状态', 'status'): return {'kind': 'status'}
    if key in ('有哪些地点', '列出地点', '地点列表'): return {'kind': 'places'}
    if key in ('你会什么', '技能列表', '有哪些技能'): return {'kind': 'skills'}
    if re.search(r'不要|不许|禁止|别做|然后|接着|再去|;|；', key): return {'kind': 'unknown'}
    for skill, entry in SKILLS.items():
        names=[entry['name']]+(['侧翻'] if skill=='sideflip' else [])
        full={prefix+name+suffix for prefix in ('做个','来个','做一个','表演一个','执行')
              for name in names for suffix in ('','动作')}
        if key in entry['aliases'] or key in full: return {'kind': 'skill', 'skill': skill}
    nav = re.fullmatch(r'(?:带我去|导航到|前往|走到|去)(.+)', key)
    if nav and nav.group(1): return {'kind': 'navigate', 'destination': nav.group(1)}
    return {'kind': 'unknown'}
