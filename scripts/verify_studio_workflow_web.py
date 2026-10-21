"""Browser regression for inline skill evidence and queue inference entry."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8765');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        b=p.chromium.launch(executable_path='/usr/local/bin/google-chrome',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1050})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        data=page.request.get(a.base+'/api/capabilities').json();catalog=page.request.get(a.base+'/api/catalog').json()
        page.goto(a.base+'/#overview',wait_until='domcontentloaded');checked=[]
        for skill in data['capabilities']:
            page.locator('.capability-progress-row').filter(has_text=skill['name']).click()
            left=page.get_by_role('region',name='工作台技能证据')
            left.get_by_role('heading',name=skill['name'],exact=True).wait_for()
            assert page.url.endswith('#overview') and page.get_by_role('dialog').count()==0
            assert left.get_by_role('link',name='查看对应能力矩阵').count()==1
            if skill['local_videos']:
                assert left.locator('video').get_attribute('src')==skill['local_videos'][0]['url']
            assert left.get_by_role('link',name='仿真手册',exact=True).count()==1
            checked.append(skill['id'])
        side=next(x for x in data['capabilities'] if x['id']=='sideflip')
        page.locator('.capability-progress-row').filter(has_text=side['name']).click()
        left=page.get_by_role('region',name='工作台技能证据');left.scroll_into_view_if_needed()
        page.screenshot(path=str(a.output/'overview.png'),full_page=True)
        left.get_by_role('link',name='查看对应能力矩阵').click()
        page.locator('#capability-sideflip.focused').wait_for()
        assert page.get_by_role('dialog').count()==0
        page.goto(a.base+'/#overview',wait_until='domcontentloaded');page.locator('.capability-progress-row').filter(has_text=side['name']).click()
        page.get_by_role('link',name='查看权重、下载与证据 →').first.click()
        page.get_by_role('heading',name='模型与证据',exact=True).wait_for()
        page.get_by_role('link',name='下载原始权重').first.wait_for()
        assert 'path=' in page.url
        page.goto(a.base+'/#simulation',wait_until='domcontentloaded')
        page.get_by_label('仿真权重来源',exact=True).select_option('queue')
        page.get_by_role('heading',name='任务队列最佳候选 · 实时仿真').wait_for()
        jobs=page.request.get(a.base+'/api/jobs').json()
        if not any(j.get('weights',{}).get('candidates') for j in jobs):
            page.get_by_role('heading',name='当前队列尚无已保存的训练权重').wait_for()
            assert page.get_by_role('button',name='启动所选权重实时仿真').is_disabled()
        page.screenshot(path=str(a.output/'queue.png'),full_page=True)
        # UI-only fixture: verifies selecting a task defaults to its selected
        # checkpoint. Actual process/physics tested separately by verify_checkpoint_live.
        mock=[dict(id='ui_fixture',name='界面测试夹具',state='snapshot',config={'task':'train','adapter':'moe','robot_id':'go2','algorithm_id':'moe_terrain_ppo','kind':'stairs'},weights={'candidates':[{'path':'outputs/studio/runs/ui_fixture/model_10.pt'},{'path':'outputs/studio/runs/ui_fixture/model_20.pt'}],'selected':{'path':'outputs/studio/runs/ui_fixture/model_10.pt'},'criterion':'保存点训练平均奖励最高','note':'UI fixture'})]
        test=b.new_page(viewport={'width':1440,'height':1050});test.on('pageerror',lambda e:errors.append(str(e)))
        test.route('**/api/jobs',lambda route:route.fulfill(json=mock))
        test.route('**/api/events',lambda route:route.fulfill(body='',content_type='text/event-stream'))
        test.goto(a.base+'/#simulation',wait_until='domcontentloaded');test.get_by_label('仿真权重来源',exact=True).select_option('queue')
        test.get_by_label('选择训练任务',exact=True).select_option('ui_fixture')
        assert test.get_by_label('在线仿真权重',exact=True).input_value().endswith('model_10.pt')
        assert test.get_by_role('button',name='启动所选权重实时仿真').is_enabled()
        test.close();assert not errors,errors
        report=dict(inline_skill_checks=checked,overview_stays_in_place=True,overview_dialogs=0,matrix_focus_without_dialog=True,
                    weight_detail_link=True,queue_source_entry=True,queue_best_selection_ui_fixture=True,js_errors=errors,
                    note='UI fixture checks selection only; real native checkpoint inference evidence is separate.')
        (a.output/'browser.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False));b.close()
if __name__=='__main__':main()
