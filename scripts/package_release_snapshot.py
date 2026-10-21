"""Collect explicit experiment resources; never archives credentials or upstream caches."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();stage=a.output.with_suffix('')
    stage.mkdir(parents=True,exist_ok=False)
    files=[];missing=[];excluded=[]
    def copy(src,dst):
        src=Path(src)
        if not src.is_file():missing.append(str(src));return
        if src.stat().st_size>95*1024**2:excluded.append(dict(path=str(src),reason='exceeds normal Git file limit; retained on server'));return
        dest=stage/dst;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        files.append(dict(path=str(dst).replace('\\','/'),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source=str(src)))
    def metadata(run,dst):
        for f in run.iterdir():
            if f.is_file() and f.suffix in ('.json','.jsonl','.yaml','.yml'):copy(f,Path(dst)/f.name)
    def weight(run,dst,model,export=None):
        copy(run/model,Path('weights')/dst/model);metadata(run,Path('weights')/dst)
        if export:
            for f in export.iterdir():
                if f.is_file():copy(f,Path('weights')/dst/'exported'/f.name)
    back=Path('/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip/logs/rsl_rl/go2_backflip/2026-10-21_00-04-59_rob2nav_reproduction')
    weight(back,'backflip/selected','model_4999.pt',back/'exported')
    for f in (back/'params').glob('*.yaml'):copy(f,Path('weights/backflip/selected/params')/f.name)
    side=root/'outputs/sideflip_dc_adapt_20261021'
    weight(side,'sideflip/selected','model_5200.pt',side/'probe_5200')
    weight(side,'sideflip/dc_final_unvalidated','model_5998.pt')
    run=root/'outputs/sideflip_native_full_20261021'
    weight(run,'sideflip/native_5000','model_4999.pt',run/'exported_final')
    run=root/'outputs/upright_walk_full_20261021'
    weight(run,'upright_walk/representative_failed_baseline','model_4999.pt',run/'exported_final')
    run=root/'outputs/upright_dc_adapt_20261021'
    weight(run,'upright_walk/dc_final_unvalidated','model_5998.pt')
    run=root/'outputs/upright_stepping_full_20261021'
    models=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    if models:weight(run,'upright_walk/stepping_snapshot_unvalidated',models[-1].name)
    for name in ('stairs','platform','multiterrain'):
        run=root/('outputs/'+name+'_reproduction');weight(run,name+'/manual_reproduction','model_199.pt')
    suffixes={'.json','.jsonl','.npz','.yaml','.yml','.log','.txt','.csv','.mp4','.png','.patch'}
    prefixes=('sideflip_','upright_','backflip_final_','navigation_live_','navigation_ppo_eval_','paw_contact_','scan_native_','scan_mujoco_web_','live_skills_web_validation','stairs_evaluation','platform_evaluation','multiterrain_evaluation')
    for run in sorted((root/'outputs').iterdir()):
        if run.is_dir() and run.name.startswith(prefixes):
            for f in sorted(run.rglob('*')):
                if f.is_file() and f.suffix in suffixes:copy(f,Path('evaluations')/run.name/f.relative_to(run))
    work=Path('/root/gpufree-data/capability-work')
    for relative in ['backflip-full.log','backflip-full.exit','backflip-final-mujoco.log','terrain-manuals.log','terrain-manuals.exit',
        'sideflip/full.log','sideflip/full.exit','sideflip/dc-adapt.log','sideflip/dc-adapt.exit',
        'upright_walk/full.log','upright_walk/full.exit','upright_walk/dc-adapt.log','upright_walk/dc-adapt.exit',
        'upright_walk/stepping-full.log','upright_walk/gait-contact-v3.log','upright_walk/gait-contact-v3.exit']:
        copy(work/relative,Path('logs')/relative)
    copy(Path('/root/gpufree-data/scan-reproduction/jazzy-compatibility.patch'),Path('source_patches/scan-jazzy-compatibility.patch'))
    copy(root/'outputs/sideflip_reference_fk_contract.json',Path('provenance/sideflip_reference_fk_contract.json'))
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=files,missing=missing,excluded=excluded,
        note='Checkpoint and log snapshot only. Selection and limitations are in README.md / model index. Never resume merely because a file exists.',hardware_connected=False)
    (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in stage.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(stage).as_posix())
    print(json.dumps(dict(archive=str(a.output),bytes=a.output.stat().st_size,files=len(files),missing=missing,excluded=excluded),indent=2),flush=True)


if __name__=='__main__':main()
