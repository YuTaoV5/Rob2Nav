import ast,copy,sys,os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from skills.backflip.launcher_patch import configure_fixstand_source
from studio.registry import registry
from studio.config import RunConfig,command


def test_peak_flag_configures_substep_history_instead_of_50hz_samples():
    source='def main():\n    env_cfg = Go2BackflipEnvCfg()\n    return env_cfg\n'
    result=configure_fixstand_source(source,'/pinned',peak_contact_forces=True)
    cfg=NS(rewards=NS(use_peak_contact_forces=False),sim=NS(dt=.005),decimation=4,
           contact_sensor=NS(update_period=.02,history_length=3))
    with patch('skills.fixstand.configure_backflip'):
        scope={'Go2BackflipEnvCfg':lambda:cfg};exec(compile(result,'<peak-contact-test>','exec'),scope)
        out=scope['main']()
    assert out.rewards.use_peak_contact_forces
    assert out.contact_sensor.update_period==out.sim.dt
    assert out.contact_sensor.history_length>=out.decimation


def test_front_web_default_and_command_include_peak_contacts(tmp_path):
    data=copy.deepcopy(next(r for r in registry()['recipes'] if r['id']=='frontflip_native_ppo')['defaults'])
    config=RunConfig.model_validate(data)
    cmd=command(config,tmp_path,tmp_path/'outputs/studio/runs/PREVIEW',tmp_path/'preview.json','/isaac/python.sh')
    assert config.recipe_options['peak_contact_forces'] is True
    assert '--peak-contact-forces' in cmd and '--fixstand-roundtrip' in cmd


def test_peak_contacts_cannot_silently_run_in_legacy_launcher_mode():
    data=copy.deepcopy(next(r for r in registry()['recipes'] if r['id']=='backflip_native_ppo')['defaults'])
    data['recipe_options'].update(peak_contact_forces=True,fixstand_roundtrip=False)
    with pytest.raises(ValueError,match='peak_contact_forces'):RunConfig.model_validate(data)


def test_launcher_preview_compiles_and_records_peak_contacts_without_training(tmp_path,monkeypatch):
    file=Path(__file__).parents[1]/'skills/backflip/train_isaac51.py'
    tree=ast.parse(file.read_text())
    last=tree.body[-1]
    assert isinstance(last,ast.Expr) and isinstance(last.value,ast.Call) and last.value.func.id=='exec'
    # Replace the final training-entry execution with a source capture. None
    # of Isaac, gym, PPO or the simulation application is started by this test.
    tree.body[-1]=ast.Assign(targets=[ast.Name(id='_rendered',ctx=ast.Store())],value=ast.Name(id='text',ctx=ast.Load()))
    source=tmp_path/'upstream';source.mkdir()
    (source/'train.py').write_text('''simulation_app = app_launcher.app
def main():
    env_cfg = Go2BackflipEnvCfg()
    agent_cfg = Go2BackflipPPORunnerCfg()
    log_dir = (log_root / run_dir_name).resolve()
    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=True)
''',encoding='utf-8')
    monkeypatch.setattr(sys,'argv',[str(file),'--source',str(source),
        '--fixstand-roundtrip','--fixstand-output',str(tmp_path/'preview_not_created'),
        '--delayed-entry','--entry-hold-scale','.1','--flip-direction','front','--peak-contact-forces'])
    monkeypatch.setattr(sys,'path',list(sys.path));monkeypatch.setattr(os,'chdir',lambda path:None)
    scope={'__file__':str(file),'__name__':'contract_preview'}
    exec(compile(ast.fix_missing_locations(tree),str(file),'exec'),scope)
    rendered=scope['_rendered'];compile(rendered,'<compiled-preview-not-executed>','exec')
    calls=[n for n in ast.walk(ast.parse(rendered)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write_contract']
    assert len(calls)==1
    keywords={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}
    assert keywords['peak_contact_forces'] is True and keywords['flip_direction']=='front'
    assert keywords['actor_observations']==60 and keywords['entry_hold_scale']==.1
    assert not (tmp_path/'preview_not_created').exists()


def test_real_upstream_peak_helper_catches_a_vanished_contact():
    torch=pytest.importorskip('torch')
    source=Path(os.environ.get('ROB2NAV_UPSTREAMS',str(Path(__file__).resolve().parents[2]/'Rob2Nav-upstreams')))/'GO2_backflip'
    if not source.is_dir():pytest.skip('Pinned upstream integration requires ROB2NAV_UPSTREAMS/GO2_backflip')
    files=list(source.rglob('go2_backflip_env.py'));assert len(files)==1
    tree=ast.parse(files[0].read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_peak_contact_force')
    method.returns=None
    for arg in method.args.args:arg.annotation=None
    scope={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'<pinned-peak-helper>','exec'),scope)
    history=torch.zeros(1,4,1,3);history[0,2,0,2]=150.
    obj=NS(cfg=NS(rewards=NS(use_peak_contact_forces=False),decimation=4),
           _contact_sensor=NS(data=NS(net_forces_w=torch.zeros(1,1,3),net_forces_w_history=history)))
    sample=scope['_peak_contact_force'](obj,torch.tensor([0]))
    obj.cfg.rewards.use_peak_contact_forces=True
    peak=scope['_peak_contact_force'](obj,torch.tensor([0]))
    assert sample.item()==0. and peak.item()==150.
