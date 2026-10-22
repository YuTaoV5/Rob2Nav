"""Prepare typed form configurations and previews; never create or submit jobs."""
from pathlib import Path
import copy,hashlib,json
from studio.config import RunConfig,command
from studio.registry import registry
from studio.recipe_training import prerequisites

root=Path('/root/gpufree-data/Rob2Nav')
out=root/'artifacts/prepared_web_stability_v2_20261022';out.mkdir(exist_ok=False)
recipes={r['id']:r for r in registry()['recipes']}
cases=[
 ('front_v2','frontflip_native_ppo','前翻v2 起跳探索',54,6000,4096,None,
  dict(fixstand_roundtrip=True,delayed_entry=True,entry_hold_scale=.1,urdf_speed_limits=True,
       finetune_learning_rate=.0003,initial_std=.7,min_action_std=.2,max_action_std=1.,peak_contact_forces=True)),
 ('side_v12','sideflip_native_ppo','侧翻v12 低学习率子步约束',55,1200,4096,
  'outputs/fixstand_sideflip_substep_peaks_v11_20261022/model_11900.pt',
  dict(transfer_motors=True,fixstand_roundtrip=True,sideflip_direction='left',motion='outputs/fixstand_finetune_20261021/sideflip.npz',
       boundary_residual=True,boundary_residual_limit=.2,boundary_bridge_limit=.05,
       boundary_terminal_hip_taper_frames=50,boundary_blend_frames=10,quiet_preparation=True,
       peak_endpoint_objective=True,landing_constraints=True,substep_objective=True,
       finetune_learning_rate=.000003,reset_std=.015)),
 ('step_from_v3','upright_walk_ppo','原地踏步v1 从直立v3',56,2000,2048,
  'outputs/fixstand_upright_tight_endpoint_v3_20261022/model_2998.pt',
  dict(transfer_motors=True,fixstand_roundtrip=True,stepping_gait=True,alternating_rear_gait=True,
       in_place_stepping=True,stationary_upright=False,endpoint_residual=True,endpoint_residual_limit=.05,
       upright_reset_fraction=.5,finetune_learning_rate=.00003,reset_std=.08)),
 ('step_from_static','upright_walk_ppo','原地踏步v1 从静态4500',57,2000,2048,
  'outputs/fixstand_upright_stand_static_v1_20261022/model_4500.pt',
  dict(transfer_motors=True,fixstand_roundtrip=True,stepping_gait=True,alternating_rear_gait=True,
       in_place_stepping=True,stationary_upright=False,endpoint_residual=True,endpoint_residual_limit=.05,
       upright_reset_fraction=.5,finetune_learning_rate=.00003,reset_std=.08)),
 ('right_v1','sideflip_native_ppo','右侧翻v1 独立参考',58,5000,4096,None,
  dict(transfer_motors=True,fixstand_roundtrip=True,sideflip_direction='right',
       motion='outputs/fixstand_sideflip_substep_peaks_v11_20261022/right_reflected/rightflip.npz',
       boundary_residual=True,boundary_residual_limit=.2,boundary_bridge_limit=.05,
       boundary_terminal_hip_taper_frames=50,boundary_blend_frames=10,quiet_preparation=True,
       peak_endpoint_objective=True,landing_constraints=True,substep_objective=True,
       finetune_learning_rate=.0003,reset_std=.3)),
]
rows=[]
for ident,rid,name,seed,iterations,num_envs,parent,options in cases:
 for mode in ('smoke','formal_after_smoke'):
  data=copy.deepcopy(recipes[rid]['defaults'])
  data.update(name=name+(' 烟测' if mode=='smoke' else ' 正式候选'),seed=seed,
   iterations=2 if mode=='smoke' else iterations,num_envs=32 if mode=='smoke' else num_envs,
   initialization='resume' if parent else 'base',checkpoint=parent or '')
  data['recipe_options'].update(options)
  cfg=RunConfig.model_validate(data)
  missing=prerequisites(cfg,root)
  if missing:raise RuntimeError('Missing reviewed dependencies: '+str(missing))
  preview=command(cfg,root,root/'outputs/studio/runs/PREVIEW',root/'outputs/studio/PREVIEW.json',
    '/root/isaacsim/python.sh','/root/gpufree-data/envs/go2-mapless/bin/python')
  filename=f'{ident}_{mode}.json'
  (out/filename).write_text(cfg.model_dump_json(indent=2)+'\n')
  rows.append(dict(id=ident,mode=mode,configuration=filename,validated=True,
   parent_checkpoint=parent,parent_sha256=hashlib.sha256((root/parent).read_bytes()).hexdigest() if parent else None,
   preview_command=preview,do_not_execute_directly=True,submitted=False))
(out/'manifest.json').write_text(json.dumps(dict(cases=rows,submitted_jobs=0,
 next_action='Use the website create-training form only after its browser origin permission is allowed. Smoke first; inspect the real config, budget and controller contract before any formal submission.',
 caveats=['Candidate hyperparameters are hypotheses; no convergence claim.',
          'Side11900 is the less-degraded diagnostic candidate, not an accepted skill.',
          'Static4500 failed both diagnostic balance seeds; it is a comparison branch, not a validated base.',
          'The formal files are proposals and must not run before their corresponding UI smoke is verified.',
          'Both left and right sides require separate final directional acceptance.'],
 training_submission_route='website new-training flow',browser_verification='pending'),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(directory=str(out),validated_form_configs=len(rows),submitted_jobs=0)),flush=True)
