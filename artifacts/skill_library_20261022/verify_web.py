import json,time,urllib.request,urllib.error,shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
base='http://127.0.0.1:8765';root=Path('/root/gpufree-data/Rob2Nav');out=root/'artifacts/skill_library_20261022';out.mkdir(parents=True,exist_ok=True)
def call(path,payload=None):
 req=urllib.request.Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Studio-Request':'1'})
 with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
def wait_for(test,seconds=60):
 deadline=time.monotonic()+seconds
 while time.monotonic()<deadline:
  s=call('/api/live-skills/state')
  if test(s):return s
  time.sleep(.25)
 raise AssertionError(s)
rows=call('/api/live-skills/library')['skills'];assert len(rows)==9 and all(x['available'] for x in rows),rows
results=[]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/local/bin/google-chrome',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1500,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(base+'/#deployment');lab=page.locator('.deployment-skill-lab');lab.locator('.deployment-skill-choice').first.wait_for()
 assert lab.locator('.deployment-skill-choice').count()==9
 lab.screenshot(path=str(out/'skill_library.png'))
 for skill,finish in [('jump',True),('backflip',True),('sideflip',True),('upright_walk',True),('stairs',False),('platform',False),('obstacles',True),('multiterrain',False)]:
  row=next(x for x in rows if x['id']==skill)
  lab.get_by_role('button',name=row['name']+' ',exact=False).click()
  lab.get_by_role('button',name='运行所选技能',exact=True).click()
  s=wait_for(lambda s:s.get('skill')==skill and (s.get('frames',0)>=5 or s['phase']=='FAILED'),60)
  assert s['phase']!='FAILED',s
  with urllib.request.urlopen(base+'/api/live-skills/frame.jpg') as r:frame=r.read()
  assert frame[:2]==b'\xff\xd8';(out/(skill+'.jpg')).write_bytes(frame)
  if finish:
   s=wait_for(lambda s:s.get('skill')==skill and s['phase'] in ('COMPLETED','FAILED'),160)
   assert s['phase']=='COMPLETED',s
   s=wait_for(lambda s:s.get('skill')==skill and bool(s.get('result')),30)
   if skill=='jump':assert s['result']['successes']==1,s
   stream=root/'outputs/live_skills'/s['id']
   shutil.copyfile(stream/'evaluation/summary.json',out/(skill+'_summary.json'))
  result=dict(skill=skill,phase=s['phase'],frames=s['frames'],time=s['time'],evaluation=s.get('result'),scope='full rollout' if finish else 'startup and live frame smoke only')
  results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
  if skill=='jump':
   page.wait_for_function("document.querySelector('.deployment-skill-lab')?.textContent.includes('实测机身抬升')")
   lab.screenshot(path=str(out/'jump_completed.png'))
  call('/api/live-skills/stop',{})
 # Both paws and a dragged virtual hand use the real, isolated contact worker.
 lab.get_by_role('button',name='伸爪握手 ',exact=False).click();lab.get_by_role('button',name='打开握手交互仿真').click()
 iframe=page.frame_locator('iframe[title="伸爪握手交互仿真"]');iframe.locator('#start').wait_for()
 def paw_wait(test):
  end=time.monotonic()+35
  while time.monotonic()<end:
   s=call('/handshake-api/api/state')
   if test(s):return s
   page.wait_for_timeout(200)
  raise AssertionError(s)
 paw_results=[]
 for leg,button in [('FR','right'),('FL','left')]:
  iframe.locator('#reset').click();paw_wait(lambda s:s['state']=='WAIT')
  iframe.locator('#'+button).click();paw_wait(lambda s:s['selected_leg']==leg and s['workspace_valid'])
  handle=iframe.locator('#handle');box=handle.bounding_box();page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2);page.mouse.down();page.mouse.move(box['x']+box['width']/2+8,box['y']+box['height']/2);page.mouse.up()
  page.wait_for_timeout(300);iframe.locator('#start').click();s=paw_wait(lambda s:s['state']=='HOLD')
  paw_results.append(dict(leg=leg,state=s['state'],force=s['force'],dragged=True))
  lab.screenshot(path=str(out/('handshake_'+leg+'.png')))
 iframe.locator('#stop').click();paw_wait(lambda s:s['state'] in ('WAIT','COOLDOWN'))
 page.set_viewport_size({'width':480,'height':1000});lab.get_by_role('button',name='原地跳高 ',exact=False).click();lab.screenshot(path=str(out/'mobile.png'))
 assert not errors,errors
 summary=dict(website_passed=True,skills=results,handshake=paw_results,page_errors=errors,visual_ground='world-fixed 0.25m grid and checker tiles, scene-only geoms',hardware_connected=False)
 (out/'web_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('ALL_WEB_CHECKS_PASSED',flush=True)
 b.close()
