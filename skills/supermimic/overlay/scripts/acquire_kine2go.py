"""Acquire a small, pinned set of public teachers; never starts training."""
import hashlib, json, os, time, urllib.request
from pathlib import Path
ROOT = Path(os.environ.get('KINE_ACQUISITION_ROOT','/root/gpufree-data/go2-supermimic/reproduction/animal_expression_v1'))
ROOT.mkdir(parents=True, exist_ok=True)
state = {'phase':'acquiring', 'training_started':False, 'artifacts':[], 'failures':[]}
def save():
    state['updated_at_unix']=time.time()
    p=ROOT/'acquisition_state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(ROOT/'acquisition_state.json')
def fetch(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        req=urllib.request.Request(url,headers={'User-Agent':'Rob2Nav-research/1.0'})
        with urllib.request.urlopen(req,timeout=35) as r,open(str(path)+'.part','wb') as f:
            while chunk:=r.read(1024*1024):f.write(chunk)
        os.replace(str(path)+'.part',path)
    return {'url':url,'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
save()
try:
    info=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/MIMUW-Robotics/kine2go',timeout=25))
    revision=info['sha'];state['hf_revision']=revision;save()
    base=f'https://huggingface.co/datasets/MIMUW-Robotics/kine2go/resolve/{revision}/'
    for rel in ['README.md','metadata.json']:
        try:state['artifacts'].append(fetch(base+rel,ROOT/'teachers'/rel))
        except Exception as e:state['failures'].append({'file':rel,'error':repr(e)})
        save()
    for clip in ['ai4_dog_trot_00','ai4_dog_pace','solo8_crawl_slow']:
        for rel in ['config.json','clip.json','motion.npy','logs/model.pt']:
            path='data/'+clip+'/'+rel
            try:state['artifacts'].append(fetch(base+path,ROOT/'teachers'/clip/rel))
            except Exception as e:state['failures'].append({'file':path,'error':repr(e)})
            save()
    state['phase']='acquired_pending_contract_review' if not state['failures'] else 'partial_download_requires_review'
except Exception as e:
    state['phase']='blocked_download';state['failures'].append({'error':repr(e)})
save();print(json.dumps(state,indent=2),flush=True)
