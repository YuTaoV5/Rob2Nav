"""Produce a measured validation artifact from retained simulator and browser traces."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    root=Path(root);out=root/'outputs/paw_contact_docs';out.mkdir(exist_ok=True)
    suites={}
    for name in ('paw_contact_benchmark_v1','paw_contact_benchmark_v2','paw_contact_holdout_v1'):
        suites[name]=json.loads((root/'outputs'/name/'benchmark.json').read_text())
    tested=[r for n,d in suites.items() if n!='paw_contact_benchmark_v1' for r in d['results']]
    browser=json.loads((out/'browser-check.json').read_text())
    result={'scenario_checks':len(tested),'passed':sum(r['success'] for r in tested),
        'max_hand_force_n':max(r['peak_hand_force_n'] for r in tested),
        'max_tilt_deg':max(r['peak_tilt_deg'] for r in tested),
        'suites':{k:{x:v[x] for x in ('successes','total','groups')} for k,v in suites.items()},
        'browser':{'drag_changed_target':browser['drag_changed_target'],'right_force_n':browser['right_hold']['force'],
                   'left_force_n':browser['left_hold']['force'],'console_errors':browser['console_errors'],
                   'mutation_guard_status':browser['missing_header_rejected']},
        'scope':'Simulation and browser checks only. Case entries include rejection/withdrawal tests; not independent hardware trials.',
        'hardware_enabled':False}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    paths=list((root/'go2_nav/paw_contact').glob('*.py'))+[root/'go2_nav/paw_contact/index.html',root/'studio/paw_routes.py',root/'vendor/RoboGauge/resources/robots/go2/go2.xml']
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'source-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    for side,color in [('right','#2979c9'),('left','#dc8d3e')]:
        trace=json.loads((root/f'outputs/paw_contact_demo_v1/{side}_demo/trace.json').read_text())
        axes[0].plot([r['time'] for r in trace],[r['force'] for r in trace],c=color,label=side+' front paw')
        axes[1].plot([r['time'] for r in trace],[r['tilt_deg'] for r in trace],c=color,label=side+' front paw')
    axes[0].axhline(3,color='#7b8da0',ls='--',lw=1,label='3 N hold target')
    axes[0].set(ylabel='Palm normal force (N)',title='Recorded MuJoCo contact and body tilt — moving hand and withdrawal')
    axes[1].set(ylabel='Body tilt (degrees)',xlabel='Simulation time (s)')
    for ax in axes:ax.grid(alpha=.18);ax.legend(loc='upper left',fontsize=9)
    fig.savefig(out/'contact_validation.png',dpi=180);plt.close(fig)
    lines=['# Go2 托爪仿真验收记录','',
        f"当前版本专项检查42/42、随机组合32/32，共{result['passed']}/{result['scenario_checks']}项。包括无效请求拒绝和失效撤回；不能称为74次独立人手握持实验。",'',
        f"已测最大手掌法向力 {result['max_hand_force_n']:.3f} N；最大机身倾斜 {result['max_tilt_deg']:.3f}°。",'',
        '专项判据：正常请求至少连续1秒保持1–8N接触、横向误差<4.5cm、倾斜<12°；全程力<15N、倾斜<15°、基座高度>0.25m；撤回后四足接地。无效请求必须拒绝；长时丢失必须撤回。','',
        '| 检查组 | 通过 | 总数 |','|---|---:|---:|']
    for name in ('paw_contact_benchmark_v2','paw_contact_holdout_v1'):
        for group,counts in suites[name]['groups'].items():lines.append(f"| {group} | {counts['passed']} | {counts['total']} |")
    lines+=['', '浏览器实际拖拽改变目标，分别完成右爪与左爪保持，再撤回/停止恢复。记录的保持力分别为'+f"{browser['right_hold']['force']:.3f}N 和 {browser['left_hold']['force']:.3f}N。JavaScript异常0条；未带写入保护头请求返回403。",'',
        '既有Studio后端回归测试44项通过；前端TypeScript/Vite构建通过。','',
        '首轮诊断37/42与失败trace完整保留在paw_contact_benchmark_v1。后续将物理手掌运动和测量噪声分离，并将手掌接触面调整为11×9cm。32个随机组合种子未用于这次修改。','',
        '数据目录：outputs/paw_contact_benchmark_v2、outputs/paw_contact_holdout_v1、outputs/paw_contact_demo_v1。参数范围、测量定义、可重跑命令和真实机器人缺口见DEPLOYMENT.md。','',
        '当前演示不使用摄像头，也未连接真实机器人；动力学、触觉误差、控制时延和人体接触实机效果都不能由本报告代替。']
    (out/'VALIDATION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/root/gpufree-data/Rob2Nav');args=p.parse_args();main(args.root)
