import argparse,asyncio,json,time,urllib.request
from pathlib import Path
from playwright.async_api import async_playwright
parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://127.0.0.1:8765');parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
BASE=args.base.rstrip('/')
def request(path,body=None):
 data=None if body is None else json.dumps(body).encode()
 return json.load(urllib.request.urlopen(urllib.request.Request(BASE+path,data=data,headers={'Content-Type':'application/json','X-Studio-Request':'1'}),timeout=10))
async def main():
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 report={'hardware_enabled':False,'skills':[]}
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/local/bin/google-chrome',headless=True,args=['--no-sandbox'])
  page=await browser.new_page(viewport={'width':1440,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto(BASE+'/#simulation');await page.get_by_label('工作台仿真技能').wait_for()
  report['options']=await page.get_by_label('工作台仿真技能').locator('option').all_text_contents()
  for skill in ['sideflip','backflip','upright_walk','stairs','platform','multiterrain']:
   await page.get_by_label('工作台仿真技能').select_option(skill)
   await page.get_by_role('button',name='部署并启动 MuJoCo',exact=True).click()
   deadline=time.monotonic()+24;s={}
   while time.monotonic()<deadline:
    s=request('/api/live-skills/state')
    if s.get('skill')==skill and s.get('frames',0)>=3:break
    if s.get('phase')=='FAILED':raise AssertionError(s)
    await asyncio.sleep(.3)
   assert s.get('frames',0)>=3,s
   image=page.get_by_alt_text('所选技能的实时 MuJoCo 物理画面');await image.wait_for();await image.scroll_into_view_if_needed()
   box=await image.bounding_box();x=box['x']+box['width']*.45;y=box['y']+box['height']*.45
   await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+95,y+25,steps=8);await page.mouse.up()
   await page.mouse.wheel(0,-180);await asyncio.sleep(.5)
   await page.mouse.move(x,y);await page.mouse.down(button='right');await page.mouse.move(x+40,y+15,steps=4);await page.mouse.up(button='right');await asyncio.sleep(.4)
   after=request('/api/live-skills/state');camera=after.get('camera',{})
   assert camera.get('azimuth',130)!=130 and camera.get('distance',3)<3 and abs(camera.get('pan_x',0))>.05,after
   report['skills'].append({'skill':skill,'id':s['id'],'frames':after['frames'],'camera':camera,'actual_backend':after.get('backend'),'note':after.get('note')})
   await page.screenshot(path=str(out/(skill+'.png')),full_page=True)
   await page.get_by_role('button',name='停止仿真',exact=True).click()
  await page.goto(BASE+'/#navigation')
  for mode in ['ppo','scan']:
   await page.get_by_label('实时导航算法').select_option(mode)
   frame=page.frame_locator('iframe[title="Go2实时MuJoCo导航"]')
   image=frame.locator('#frame');await image.wait_for();await image.scroll_into_view_if_needed();await asyncio.sleep(1)
   box=await image.bounding_box();x=box['x']+box['width']/2;y=box['y']+box['height']/2
   await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+80,y+30,steps=6);await page.mouse.up();await page.mouse.wheel(0,-180);await asyncio.sleep(.7)
   state=request(('/scan-live-api' if mode=='scan' else '/navigation-live-api')+'/api/state')
   assert state['camera']['azimuth']!=135 and state['camera']['distance']<5,state
   report[mode+'_camera']=state['camera']
  await page.goto(BASE+'/handshake');image=page.locator('#scene');await image.wait_for();await asyncio.sleep(1)
  await image.scroll_into_view_if_needed();box=await image.bounding_box();x=box['x']+100;y=box['y']+100
  await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+70,y+15,steps=5);await page.mouse.up();await asyncio.sleep(.6)
  hand=request('/handshake-api/api/state');assert hand['camera_settings']['azimuth']!=125,hand
  report['handshake_camera']=hand['camera_settings']
  # A finished rollout remains inspectable without more physics stepping.
  request('/api/live-skills/start',{'skill':'sideflip'})
  deadline=time.monotonic()+25
  while time.monotonic()<deadline:
   finished=request('/api/live-skills/state')
   if finished.get('phase')=='COMPLETED':break
   await asyncio.sleep(.3)
  assert finished['phase']=='COMPLETED',finished
  before_time=finished['time'];request('/api/live-skills/camera',{'azimuth':44,'distance':2,'elevation':-30})
  await asyncio.sleep(.5);inspected=request('/api/live-skills/state')
  assert inspected['time']==before_time and inspected['camera']['azimuth']==44,inspected
  report['completed_state_camera']={'time':before_time,'camera':inspected['camera']}
  request('/api/live-skills/stop',{})
  report['console_errors']=errors;assert not errors,errors
  report['passed']=True
  (out/'browser.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
  await browser.close()
asyncio.run(main())
