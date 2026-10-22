import json,time,urllib.request,shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path('/root/gpufree-data/Rob2Nav');out=root/'artifacts/skill_library_20261022';base='http://127.0.0.1:8765'
def call(path,payload=None):
 req=urllib.request.Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Studio-Request':'1'})
 with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/local/bin/google-chrome',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1500,'height':1100});page.goto(base+'/#deployment');lab=page.locator('.deployment-skill-lab')
 lab.get_by_role('button',name='原地跳高 ',exact=False).click();lab.get_by_role('button',name='运行所选技能',exact=True).click()
 end=time.monotonic()+60;peak=0.
 while time.monotonic()<end:
  s=call('/api/live-skills/state')
  if s.get('skill')=='jump' and s.get('frames') and s.get('position',[0,0,0])[2]>peak:
   peak=s['position'][2]
   with urllib.request.urlopen(base+'/api/live-skills/frame.jpg') as r:(out/'jump_peak.jpg').write_bytes(r.read())
  if s.get('result') and s['phase']=='COMPLETED':break
  if s['phase']=='FAILED':raise AssertionError(s)
  page.wait_for_timeout(100)
 else:raise AssertionError(s)
 assert s['result']['successes']==1 and s['result']['hardware_constraints_pass'],s
 run=root/'outputs/live_skills'/s['id'];shutil.copyfile(run/'evaluation/summary.json',out/'jump_summary.json');shutil.copyfile(run/'evaluation/trace.json',out/'jump_trace.json')
 report=json.loads((out/'web_validation.json').read_text());shutil.copyfile(out/'web_validation.json',out/'initial_web_validation.json')
 for x in report['skills']:
  if x['skill']=='jump':x.update(phase=s['phase'],time=s['time'],frames=s['frames'],evaluation=s['result'],scope='full rollout, retested after reference slew limiting')
 page.wait_for_function("document.querySelector('.deployment-skill-lab')?.textContent.includes('本次仿真电机约束：通过')")
 lab.screenshot(path=str(out/'jump_completed.png'));lab.locator('img.composer-frame').scroll_into_view_if_needed();box=lab.locator('img.composer-frame').bounding_box()
 page.mouse.move(box['x']+100,box['y']+100);page.mouse.down();page.mouse.move(box['x']+180,box['y']+120);page.mouse.up();page.wait_for_timeout(500)
 after=call('/api/live-skills/state');assert after['camera']['azimuth']!=130 and after['time']==s['time'],after
 report.update(camera_drag_verified=True,final_jump_retested=True)
 (out/'web_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 page.set_viewport_size({'width':480,'height':1000});lab.screenshot(path=str(out/'mobile.png'));b.close()
 print(json.dumps(report['skills'][0],ensure_ascii=False),flush=True)
# Verify the existing continuous sequence still works with the visual overlay.
source=(root/'artifacts/composer_timing_20261022/verify_web.py').read_text().replace('artifacts/composer_timing_20261022','artifacts/skill_library_20261022/composition')
exec(compile(source,'verify_composer_timing.py','exec'))
