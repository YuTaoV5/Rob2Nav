import json,time,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
base='http://127.0.0.1:8765';out=Path('/root/gpufree-data/Rob2Nav/artifacts/scan_step_door_20261022/web');out.mkdir(parents=True,exist_ok=True)
def call(path,payload=None):
 req=urllib.request.Request(base+'/scan-live-api'+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Studio-Request':'1'})
 with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/local/bin/google-chrome',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1500,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(base+'/scan-live');page.locator('#scene').wait_for();assert page.locator('#scene option').count()==10
 page.select_option('#scene','stairs_door');page.click('#reset');page.wait_for_function("state?.scene==='stairs_door' && state?.phase==='READY'",timeout=30000)
 assert '5厘米' in page.locator('#terrain-note').inner_text();assert '0.95米' in page.locator('#terrain-note').inner_text()
 # An actual click selects a new goal beyond the door and all steps.
 point=page.evaluate("""()=>{const m=document.querySelector('#map');const p=m.createSVGPoint();p.x=6.3;p.y=1.2;const s=p.matrixTransform(m.getScreenCTM());return {x:s.x,y:s.y}}""")
 page.mouse.click(point['x'],point['y']);page.wait_for_function("Math.abs(state.goal[0]-6.3)<.05 && Math.abs(state.goal[1]+1.2)<.05")
 page.click('#camera-reset');page.wait_for_function("state?.camera.azimuth===135")
 box=page.locator('#frame').bounding_box();page.mouse.move(box['x']+200,box['y']+180);page.mouse.down();page.mouse.move(box['x']+240,box['y']+195);page.mouse.up();page.wait_for_function("state?.camera.azimuth!==135")
 page.screenshot(path=str(out/'door_ready.png'),full_page=True)
 page.click('#start');page.wait_for_function("state?.position[0]>2.9 && state?.position[0]<4.1",timeout=100000)
 page.screenshot(path=str(out/'on_steps_at_door.png'),full_page=True)
 page.wait_for_function("state?.phase==='ARRIVED'||state?.phase==='FAULT'||(state?.phase==='STOPPED'&&state?.reason==='navigation_timeout')",timeout=150000)
 s=call('/api/state');assert s['phase']=='ARRIVED' and s['course_metrics']['door_crossed'] and s['course_metrics']['steps_exited'],s
 page.screenshot(path=str(out/'custom_goal_arrived.png'),full_page=True)
 assert not s['collision'] and not s['fallen'];assert page.locator('#door-mark path').count()==1
 result={'clicked_goal':s['goal'],'intended_goal':[6.3,-1.2],'arrived':True,'collision':s['collision'],'fallen':s['fallen'],'course':s['course_metrics'],'scene_options':10,'camera_drag':True,'page_errors':errors}
 call('/api/control',{'action':'stop'})
 # Existing following task still loads, with its three actors and selected identity.
 page.select_option('#scene','follow_mixed');page.select_option('#person','person_02');page.click('#reset');page.wait_for_function("state?.scene==='follow_mixed'&&state?.selected_person==='person_02'&&state?.phase==='READY'",timeout=30000)
 assert page.locator('#people circle').count()==3;result['legacy_follow_load']=True
 page.select_option('#scene','stairs_door');page.click('#reset');page.wait_for_function("state?.scene==='stairs_door'&&state?.phase==='READY'",timeout=30000)
 page.set_viewport_size({'width':480,'height':1000});page.screenshot(path=str(out/'mobile.png'),full_page=True)
 assert not errors,errors;(out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
 b.close()
