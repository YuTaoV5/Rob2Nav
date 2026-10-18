"""Read-only, provenance-aware import of existing experiment artifacts."""
import hashlib
import json
import math
from pathlib import Path
import time


def wilson(success,total):
    if not total or not 0<=success<=total:return None
    z=1.96;p=success/total;denom=1+z*z/total
    centre=(p+z*z/(2*total))/denom
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [max(0,centre-radius),min(1,centre+radius)]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def ident(path):
    return hashlib.sha256(str(path).encode()).hexdigest()[:16]


def last_iteration(path):
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-40000))
        for line in reversed(f.read().splitlines()):
            try:
                d=json.loads(line)
                if isinstance(d,dict) and 'iteration' in d:return d
            except (ValueError,UnicodeDecodeError):continue
    return {}


def normalize_evaluation(d):
    if not isinstance(d,dict) or 'successes' not in d:return None
    d=dict(d);episodes=d.get('episodes')
    if 'total' not in d:
        if isinstance(episodes,list):d['total']=len(episodes)
        elif isinstance(episodes,int):d['total']=episodes
        else:return None
    if not isinstance(episodes,list):d['episodes']=d.get('results',[])
    d.setdefault('args',d.get('arguments',{}))
    return d


class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = {}
        self.data = None
        self.updated = 0.
        self.scalar_cache = {}

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def get(self, force=False):
        if self.data is not None and not force and time.monotonic()-self.updated < 30:
            return self.data
        evaluations, runs, reports, videos, models, issues = [], [], [], [], [], []
        records = {}
        roots = [p for p in (self.root/'artifacts', self.root/'outputs') if p.exists()]
        # Include live server training trees without exposing arbitrary host files.
        live = self.root.parent/'MoE-runs'
        scan_roots = roots + ([live] if live.exists() else [])
        for base in scan_roots:
            for path in base.rglob('summary.json'):
                if 'source' in path.parts or path.stat().st_size > 20_000_000:
                    continue
                d = normalize_evaluation(read_json(path))
                if d is None:
                    continue
                args = d.get('args', {}); folder = path.parent
                key = ident(path); records[key] = path
                rel = str(folder.relative_to(self.root)) if folder.is_relative_to(self.root) else 'live/'+folder.name
                course = args.get('course', 'original')
                geometry = d.get('geometry', '')
                precursor = course == 'wide_deep' or 'PRECURSOR' in geometry or args.get('shallow_probe', False)
                teacher=args.get('teacher_map') or 'teacher_' in rel.lower() or '/teacher/' in rel.replace('\\','/')
                student='student' in rel.lower() or ('teacher_map' in args and not args['teacher_map'] and bool(args.get('checkpoint')))
                role = '特权教师' if teacher else ('感知学生' if student else '历史对照')
                assisted = 'active' in rel or 'ik' in rel.lower() or 'jump_backend' in args
                video = folder/'episode_0.mp4'
                row = {'id':key,'name':folder.name,'group':str(folder.parent.name),'path':rel.replace('\\','/'),
                       'kind':args.get('kind', d.get('kind', folder.name)), 'difficulty':args.get('difficulty'),
                       'successes':d['successes'],'total':d['total'],'seed':args.get('seed'),
                       'interval95':wilson(d['successes'],d['total']),
                       'role':role,'geometry':'训练前置' if precursor else ('原AME' if 'ame_' in folder.name else '标准/历史场景'),
                       'backend':d.get('backend','MuJoCo' if assisted or 'mujoco' in rel.lower() or 'results' in d else 'Isaac / 需核对'),
                       'assistance':'技能调度 / IK' if assisted else '关节策略 / 见来源',
                       'diagnostic':bool(args.get('map_ablation','none') != 'none' or d.get('diagnostic_only')),
                       'checkpoint':d.get('checkpoint_sha256'),'video':self.relative(video) if video.exists() and video.is_relative_to(self.root) else None}
                evaluations.append(row)
            for path in base.rglob('iterations.jsonl'):
                if 'source' in path.parts: continue
                folder=path.parent; key=ident(path); records[key]=path
                status=read_json(folder/'status.json') or last_iteration(path)
                complete=read_json(folder/'completed.json') or read_json(folder/'train_summary.json')
                failed=read_json(folder/'failed.json'); manifest=read_json(folder/'manifest.json') or {'args':read_json(folder/'arguments.json',{})}
                state='failed' if failed else ('completed' if complete else 'snapshot')
                runs.append({'id':key,'name':folder.name,'iteration':status.get('iteration'),
                             'state':state,'phase':status.get('phase'),'steps_per_s':status.get('steps_per_s'),
                             'args':manifest.get('args', {}),'error':failed,'updated':path.stat().st_mtime,
                             'live':folder.is_relative_to(live) if live.exists() else False})
            if base.is_relative_to(self.root):
                for path in base.rglob('*.mp4'):
                    if 'source' in path.parts: continue
                    key=ident(path); records[key]=path
                    videos.append({'id':key,'name':path.parent.name,'path':self.relative(path),'bytes':path.stat().st_size})
                for path in base.rglob('*.pt'):
                    if path.name=='actor.pt' or 'source' in path.parts: continue
                    key=ident(path); records[key]=path
                    models.append({'id':key,'name':path.name,'group':path.parent.name,'path':self.relative(path),'bytes':path.stat().st_size})
        report_paths=list(self.root.glob('*.md'))+list((self.root/'docs').rglob('*.md'))
        if (self.root/'studio/README.md').exists():report_paths.append(self.root/'studio/README.md')
        for path in sorted(report_paths):
            key=ident(path); records[key]=path
            text=path.read_text(encoding='utf-8-sig', errors='replace')
            title=next((s.lstrip('# ').strip() for s in text.splitlines() if s.startswith('# ')),path.stem)
            reports.append({'id':key,'name':title,'path':self.relative(path),'bytes':path.stat().st_size})
        self.records=records
        self.data=clean({'evaluations':evaluations,'runs':sorted(runs,key=lambda r:r['updated'],reverse=True),
                         'reports':reports,'videos':videos,'models':models,'issues':issues,'indexed_at':time.time()})
        self.updated=time.monotonic()
        return self.data

    def path(self, key):
        self.get()
        if key not in self.records: raise KeyError(key)
        return self.records[key]

    def series(self,key,limit=2000):
        path=self.path(key)
        if path.name!='iterations.jsonl': raise ValueError('Not a training series')
        rows=[]
        with path.open(encoding='utf-8-sig',errors='replace') as f:
            for line in f:
                try:
                    row=json.loads(line)
                    if isinstance(row,dict): rows.append(row)
                except ValueError: continue  # incomplete live JSONL line
        # Select uniform points while always retaining the final iteration.
        stride=max(1,math.ceil(len(rows)/max(limit-1,1)))
        sampled=rows[::stride]
        if rows and (not sampled or sampled[-1] is not rows[-1]):sampled.append(rows[-1])
        return clean({'rows':sampled,'raw_count':len(rows),'sampled':len(sampled)<len(rows)})

    def scalars(self,key,tag=None):
        path=self.path(key)
        if path.name!='iterations.jsonl':raise ValueError('Not a training run')
        events=list(path.parent.glob('events.out.tfevents.*'))
        if not events:return {'tags':[],'points':[],'tag':None,'note':'未归档TensorBoard事件文件'}
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:return {'tags':[],'points':[],'tag':None,'note':'安装studio/requirements.txt中的tensorboard以读取奖励曲线'}
        fingerprint=tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(events))
        cached=self.scalar_cache.get(key)
        if cached is None or cached[0]!=fingerprint:
            accumulator=EventAccumulator(str(path.parent),size_guidance={'scalars':0});accumulator.Reload()
            self.scalar_cache[key]=(fingerprint,accumulator)
        else:accumulator=cached[1]
        tags=accumulator.Tags().get('scalars',[])
        chosen=tag if tag in tags else next((t for t in tags if t=='Train/mean_reward'),tags[0] if tags else None)
        values=accumulator.Scalars(chosen) if chosen else []
        stride=max(1,math.ceil(len(values)/1999));sample=values[::stride]
        if values and sample[-1] is not values[-1]:sample.append(values[-1])
        return clean({'tags':tags,'tag':chosen,'points':[[v.step,v.value] for v in sample],'raw_count':len(values),
                      'note':'TensorBoard原始标量；奖励项可能已按episode时长归一化，依来源实现解释。'})
