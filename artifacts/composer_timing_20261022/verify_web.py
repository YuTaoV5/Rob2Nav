import json,time,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('/root/gpufree-data/Rob2Nav/artifacts/composer_timing_20261022');out.mkdir(parents=True,exist_ok=True)
base='http://127.0.0.1:8765'
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/local/bin/google-chrome',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport=dict(width=1500,height=1150));errors=[];samples=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 def response(r):
  if r.url.endswith('/api/live-skills/state'):
   try:
    s=r.json()
    if s.get('skill')=='composition' and s.get('progress'):samples.append(s)
   except Exception:pass
 page.on('response',response)
 page.goto(base+'/#deployment')
 section=page.locator('.unified-composer');section.wait_for()
 section.get_by_role('button',name='＋ 日常行走（推荐）').wait_for()
 assert section.get_by_role('button',name='＋ 站稳不动').count()==1
 assert '保持四脚站立' in section.inner_text()
 cycles=section.get_by_text('循环轮数',exact=False).locator('input')
 cycles.fill('2');assert '12.0 秒' in section.locator('.composer-plan-note').inner_text()
 cycles.fill('1');assert '7.0 秒' in section.locator('.composer-plan-note').inner_text()
 # Research actions must not present a duration input that the sequencer ignores.
 section.get_by_text('显示研究技能',exact=True).locator('..').locator('input').check()
 section.get_by_role('button',name='＋ 侧空翻（研究）').click()
 flip=section.locator('.composer-step').filter(has_text='侧空翻')
 assert flip.locator('input[type=number]').count()==0
 assert '完成一次动作后继续' in flip.inner_text()
 flip.get_by_role('button',name='移除第3步').click()
 section.get_by_text('显示研究技能',exact=True).locator('..').locator('input').uncheck()
 section.screenshot(path=str(out/'before_run.png'))
 section.get_by_role('button',name='运行序列',exact=True).click()
 page.wait_for_function("document.querySelector('.composer-timing')?.textContent.includes('序列已完成') && document.querySelector('.unified-composer')?.textContent.includes('本次序列判定')",timeout=180000)
 final=json.load(urllib.request.urlopen(base+'/api/live-skills/state'))
 assert final['result']['passed'],final
 q=final['progress'];assert abs(q['planned_dwell_s']-7)<1e-8 and abs(q['dwell_elapsed_s']-7)<1e-8,q
 assert abs(q['total_elapsed_s']-q['dwell_elapsed_s']-q['extra_elapsed_s'])<1e-7,q
 assert q['extra_elapsed_s']>1 and q['initialization_s']>0,q
 assert all(abs(s['progress']['total_elapsed_s']-s['progress']['dwell_elapsed_s']-s['progress']['extra_elapsed_s'])<1e-7 for s in samples)
 assert any(s['progress']['status']=='transition' for s in samples)
 assert any(s['progress']['status']=='dwell' for s in samples)
 assert any(s['progress']['step_index']==1 and s['progress']['step_target_s']==3 for s in samples)
 timing=section.locator('.composer-timing').inner_text()
 section.screenshot(path=str(out/'completed.png'))
 # Camera remains usable after completion and never advances the simulation.
 image=section.locator('img.composer-frame');image.scroll_into_view_if_needed();box=image.bounding_box()
 page.mouse.move(box['x']+100,box['y']+100);page.mouse.down();page.mouse.move(box['x']+170,box['y']+130);page.mouse.up()
 page.wait_for_function("fetch('/api/live-skills/state').then(r=>r.json()).then(s=>s.camera.azimuth!==130)")
 after=json.load(urllib.request.urlopen(base+'/api/live-skills/state'))
 assert after['progress']['total_elapsed_s']==q['total_elapsed_s']
 page.set_viewport_size(dict(width=480,height=1000));page.reload();page.locator('.composer-timing progress').wait_for()
 page.locator('.unified-composer').screenshot(path=str(out/'mobile.png'))
 assert not errors,errors
 summary=dict(passed=True,sequence=['stand 2s','walk_moe 3s','stand 2s'],timing=q,ui_timing=timing,samples=len(samples),camera_preserves_clock=True,page_errors=errors,hardware_executed=False)
 (out/'web_smoke.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 (out/'timing_samples.json').write_text(json.dumps([dict(time=s['time'],progress=s['progress']) for s in samples],indent=2))
 print(json.dumps(summary,ensure_ascii=False,indent=2))
 b.close()
