"""Physics-based falsifiable tests for paw placement, tracking and withdrawal."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
try:
    from .handshake import PawSimulation, HandObservation
except ImportError:
    from handshake import PawSimulation, HandObservation


def cases():
    result = []
    for side in (-1,1):
        for kind in ('nominal','position','moving','noisy_delayed','brief_dropout','push','low_friction','heavy'):
            for seed in range(3 if kind=='position' else 2):
                rng=np.random.default_rng(1000+seed+100*(side+1))
                target=[float(rng.uniform(.315,.43)),side*float(rng.uniform(.085,.19)),float(rng.uniform(.145,.24))] if kind=='position' else [.37,side*.13,.19]
                result.append(dict(name=f'{"left" if side==1 else "right"}_{kind}_{seed}',kind=kind,seed=seed,target=target,
                    friction=.45 if kind=='low_friction' else .8,mass_scale=1.15 if kind=='heavy' else 1.))
        for kind in ('invalid_far','closed_palm','low_confidence','long_dropout'):
            result.append(dict(name=f'{"left" if side==1 else "right"}_{kind}',kind=kind,seed=0,target=[.60,side*.13,.19] if kind=='invalid_far' else [.37,side*.13,.19],friction=.8,mass_scale=1.))
    return result


def run(case,root,output,video=False):
    folder=Path(output)/case['name'];folder.mkdir(parents=True,exist_ok=False)
    s=PawSimulation(root,seed=case['seed'],friction=case['friction'],mass_scale=case['mass_scale'])
    s.requested_leg=0 if case['target'][1]>0 else 1
    rng=np.random.default_rng(case['seed']+17)
    rows=[];hold_current=0.;hold_best=0.;push_done=False;writer=None
    if video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(folder/'episode_0.mp4',fps=20,codec='libx264',quality=8)
    observations=[]
    for i in range(1450):
        t=float(s.data.time)
        target=np.array(case['target'],dtype=float)
        visible=1.8<t<10.5
        kind=case['kind']
        if kind in ('moving','mixed') and t>5:
            target += [0,.022*np.sin((t-5)*2),.016*np.sin((t-5)*1.6)]
        # Environment ground truth is used to move the human hand, never as a controller observation.
        s.environment_hand = (target.copy(),visible)
        if kind in ('noisy_delayed','mixed'):
            target += rng.normal(0,.003 if kind=='noisy_delayed' else .002,3)
        if kind=='brief_dropout' and 6.3<t<6.46:visible=False
        if kind=='long_dropout' and t>6.5:visible=False
        if i%3==0:
            observations.append((t,HandObservation(target.tolist(),visible=visible,open_palm=kind!='closed_palm',confidence=.5 if kind=='low_confidence' else 1.,sequence=i)))
        delay=.09 if kind=='noisy_delayed' else (.06 if kind=='mixed' else 0.)
        while observations and observations[0][0]<=t-delay:
            _,obs=observations.pop(0);s.observe(obs)
        if kind in ('push','mixed') and 7.0<t<7.2:
            s.data.xfrc_applied[s.base,:3]=[8., -12.*np.sign(target[1]),0] if kind=='push' else [4., -6.*np.sign(target[1]),0]
        else:s.data.xfrc_applied[s.base,:]=0
        s.step()
        snap=s.snapshot();snap['target']=target.tolist()
        if snap['state']=='HOLD' and 1.<=snap['force']<=8. and np.linalg.norm(np.array(snap['paw'])[:2]-np.array(snap['hand'])[:2])<.045 and snap['tilt_deg']<12:
            hold_current+=.01;hold_best=max(hold_best,hold_current)
        else:hold_current=0.
        if i%5==0:
            rows.append(snap)
            if writer:
                from PIL import Image,ImageDraw
                frame=Image.fromarray(s.render());draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,960,48),fill=(17,24,39));draw.text((16,16),f'{case["name"]} | {snap["state"]} | palm contact {snap["force"]:.2f} N | tilt {snap["tilt_deg"]:.1f} deg',fill='white')
                writer.append_data(np.asarray(frame))
        if not np.isfinite(s.data.qpos).all() or s.tilt>.8 or s.data.qpos[2]<.14:break
    if writer:writer.close()
    not_applicable=case['kind'] in ('invalid_far','closed_palm','low_confidence')
    max_tilt=max(r['tilt_deg'] for r in rows)
    return_ok=s.state in ('WAIT','COOLDOWN') and np.count_nonzero(s.ground_force>5)>=4
    stable=max_tilt<15 and min(r['base'][2] for r in rows)>.25
    placed=hold_best>=1.
    if case['kind']=='long_dropout':
        placed=any(e['state']=='HOLD' for e in s.events)
    rejected=not any(e['state'] in ('SHIFT','LIFT','HOLD') for e in s.events)
    result=dict(case=case,success=bool((rejected if not_applicable else placed) and stable and s.peak_force<15. and return_ok),
        placement=placed,rejected=rejected,stable=stable,returned=bool(return_ok),max_continuous_gentle_contact_s=hold_best,
        peak_hand_force_n=s.peak_force,peak_tilt_deg=max_tilt,final=s.snapshot(),events=s.events,
        contact_dynamics=True,criterion='>=1s continuous HOLD, 1..8N palm contact, <4.5cm lateral error, <12deg tilt; overall peak<15N, tilt<15deg, four-foot return. Invalid requests must be rejected.')
    (folder/'trace.json').write_text(json.dumps(rows))
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    s.close()
    return result


def job(args):
    return run(*args)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/root/gpufree-data/Rob2Nav');p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--video',action='store_true');p.add_argument('--mixed',action='store_true')
    args=p.parse_args();Path(args.output).mkdir(parents=True,exist_ok=False)
    testcases=cases() if not args.video else [dict(name=f'{side}_demo',kind='moving',seed=42,target=[.37,.13 if side=='left' else -.13,.19],friction=.8,mass_scale=1.) for side in ('left','right')]
    if args.mixed:
        testcases=[]
        for seed in range(32):
            rng=np.random.default_rng(9000+seed);side=-1 if seed%2==0 else 1
            testcases.append(dict(name=f'mixed_{seed:02}',kind='mixed',seed=9000+seed,
                target=[float(rng.uniform(.32,.42)),side*float(rng.uniform(.10,.18)),float(rng.uniform(.15,.23))],
                friction=float(rng.uniform(.5,1.)),mass_scale=float(rng.uniform(.85,1.15))))
    results=[]
    if args.video:
        for c in testcases:results.append(run(c,args.root,args.output,True))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(job,[(c,args.root,args.output,False) for c in testcases]):
                results.append(r);print(r['case']['name'],r['success'],round(r['peak_hand_force_n'],2),round(r['peak_tilt_deg'],2),round(r['max_continuous_gentle_contact_s'],2),flush=True)
    grouped={}
    for r in results:
        group=grouped.setdefault(r['case']['kind'],dict(passed=0,total=0))
        group['total']+=1;group['passed']+=int(r['success'])
    summary=dict(successes=sum(r['success'] for r in results),total=len(results),groups=grouped,results=results,
        scope='MuJoCo free-base rigid contact, tracked virtual palm. No camera or real robot validation.')
    (Path(args.output)/'benchmark.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)
