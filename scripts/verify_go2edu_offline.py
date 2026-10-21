"""Offline deployment audit; never imports ROS or Unitree SDK."""
import argparse,ast,hashlib,importlib.util,json,re,subprocess,sys,urllib.request,os
from pathlib import Path
from xml.etree import ElementTree as ET

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--skip-pytest',action='store_true');p.add_argument('--driver-msg-cache',type=Path);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);repo=a.repo.resolve()
git_env=dict(os.environ,GIT_CONFIG_GLOBAL=os.devnull,GIT_CONFIG_NOSYSTEM='1')
pins={'lio':('xiaofan4122/Elevator-LIO','1d79af77f3d9747ea57ef52a9b01d326a8ec561a',['yaml','msg','launch']),
      'scan':('wuyi2121/SCAN-Planner','348e8a590a50a5a6bbab8d8c6dcfd171f009be26',['src/planner/plan_manage/launch','src/planner/plan_env/src'])}
for name,(url,pin,paths) in pins.items():
 dst=a.out/name
 if not dst.exists():
  subprocess.run(['git','clone','--filter=blob:none','--no-checkout','--depth','1','https://github.com/'+url+'.git',str(dst)],check=True,env=git_env,timeout=120)
  subprocess.run(['git','-C',str(dst),'sparse-checkout','set',*paths],check=True,env=git_env,timeout=120)
  subprocess.run(['git','-C',str(dst),'fetch','origin',pin,'--depth','1'],check=True,env=git_env,timeout=120)
  subprocess.run(['git','-C',str(dst),'checkout','--detach',pin],check=True,env=git_env,timeout=120)
 assert subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()==pin
spec=importlib.util.spec_from_file_location('prepare',repo/'deployment/go2edu/prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
synthetic=dict(reviewed=True,robot='SYNTHETIC OFFLINE ONLY',initial_base_height_m=.35,T_base_lidar=[[1,0,0,.2],[0,1,0,0],[0,0,1,.15],[0,0,0,1]],T_imu_lidar=[[1,0,0,-.011],[0,1,0,-.02329],[0,0,1,.04412],[0,0,0,1]])
cal=a.out/'synthetic_not_for_hardware.json';cal.write_text(json.dumps(synthetic))
m.generate(cal,a.out/'lio',a.out/'scan',a.out/'generated')
launch=ET.parse(a.out/'generated/scan_go2edu.launch').getroot()
advanced=ET.parse(a.out/'scan/src/planner/plan_manage/launch/advanced_param.xml').getroot()
passed={x.attrib['name']:x.attrib['value'] for x in launch.find('include').findall('arg')}
accepted={x.attrib['name'] for x in advanced.findall('arg')}
assert not passed.keys()-accepted
assert all(x.attrib['name'] in passed for x in advanced.findall('arg') if 'default' not in x.attrib)
params={}
all_args={x.attrib['name']:x.attrib['default'] for x in launch.findall('arg') if 'default' in x.attrib};all_args.update(passed)
def value(v):return re.sub(r'\$\(arg ([^)]+)\)',lambda x:all_args[x[1]],v)
for node in advanced.findall('node'):
 for x in node.findall('param'):params[node.attrib['name']+'/'+x.attrib['name']]=value(x.attrib['value'])
for x in advanced.findall('param'):params[x.attrib['name']]=value(x.attrib['value'])
for x in launch.findall('param'):params[x.attrib['name']]=value(x.attrib['value'])
assert params['scan_planner_node/grid_map/need_extrinsic']=='false'
assert params['scan_planner_node/grid_map/cloud_is_world']=='true'
assert params['scan_planner_node/grid_map/frame_id']=='nav_odom'
assert float(params['closed_loop_controller/max_vx'])==.2
assert float(params['closed_loop_controller/max_vy'])==.1
assert float(params['closed_loop_controller/max_vyaw'])==.3
assert all('sim' not in node.attrib['type'] for node in launch.findall('node')+advanced.findall('node'))
def schema(text):return [' '.join(s.split()) for line in text.splitlines() if (s:=line.split('#')[0].strip())]
wire={}
for name in ('CustomMsg','CustomPoint'):
 if a.driver_msg_cache:raw=(a.driver_msg_cache/(name+'.msg')).read_text()
 else:raw=urllib.request.urlopen('https://raw.githubusercontent.com/Livox-SDK/livox_ros_driver2/21445540f0d100dc86a7e6df312dd70bbdb4afdf/msg/'+name+'.msg',timeout=30).read().decode()
 assert schema(raw)==schema((a.out/'lio/msg'/(name+'.msg')).read_text()),name
 wire[name]=True
files=list((repo/'deployment/go2edu').glob('*.py'))
for file in files:ast.parse(file.read_text(),feature_version=(3,8))
doc=(repo/'docs/GO2_EDU_NOETIC_FOXY_NAVIGATION.md').read_text()
blocks=re.findall(r'```bash\n(.*?)```',doc,re.S)
for i,block in enumerate(blocks):
 path=a.out/('manual_block_'+str(i)+'.sh');path.write_text(block)
 subprocess.run(['bash','-n',str(path)],check=True)
for name in ('open_noetic.sh','env.noetic'):subprocess.run(['bash','-n',str(repo/'deployment/go2edu'/name)],check=True)
if not a.skip_pytest:
 test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_go2edu_deployment.py','--junitxml='+str(a.out/'pytest.xml')],cwd=repo,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (a.out/'pytest.log').write_text(test.stdout);print(test.stdout);assert test.returncode==0
report=dict(scope='Offline only; no ROS/DDS/robot connection',hardware_executed=False,pytest_in_this_process=not a.skip_pytest,noetic_native_build='NOT RUN: validation host has Jazzy, not Noetic',pins={k:v[1] for k,v in pins.items()},wire_field_schemas_identical=wire,python_38_syntax_files=len(files),bash_manual_blocks_checked=len(blocks),statically_expanded_launch_parameters=params,source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[repo/'docs/GO2_EDU_NOETIC_FOXY_NAVIGATION.md']})
(a.out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
