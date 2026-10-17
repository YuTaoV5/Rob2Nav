"""Optional structural equivalence check against a local upstream checkout.

Executes the two upstream terrain functions, without their Isaac decorators.
Only use a checkout you intend to execute. No upstream source is redistributed.
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from perceptive_rl.ame_terrain import heightfield

p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
source=Path(a.upstream)/'source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/terrains/loco_hf_terrains.py'
tree=ast.parse(source.read_text(encoding='utf-8'));records=[]
for kind,name,gaps,columns in [('ame_stakes','double_column_stakes_terrain',(.1,.3),(.1,.1)),('ame_alternate','alternate_column_stakes_terrain',(.05,.15),(0.,.2))]:
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[];node.returns=None
    for arg in node.args.args:arg.annotation=None
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
    cfg=SimpleNamespace(size=(8.,8.),horizontal_scale=.05,vertical_scale=.005,stake_side_range=(.2,.4),stake_gap_range=gaps,column_gap_range=columns,column_jitter=0.,stake_height_max=.03,holes_depth=-2.,platform_width=2.,seed=42)
    for d in (0.,.5,1.):
        original=namespace[name](d,cfg)*.005;ours=heightfield(d,kind)
        equal=np.array_equal(original>-2,ours>-2)
        assert equal,(kind,d)
        records.append({'kind':kind,'difficulty':d,'support_mask_identical':bool(equal)})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'upstream_revision':'1d3519ee946c846c1c89dd660539a224f99972b5',
    'upstream_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'scope':'Exact support masks only; random top heights differ; independent box collision mesh.',
    'checks':records},indent=2))
print('SUPPORT_MASKS_IDENTICAL',len(records))
