import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_agent.places import Places
from voice_agent.intents import parse
from voice_agent.backends import MockBackend
from voice_agent.controller import Controller
from voice_agent.router import RobotRouter

ROOT=Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def advance(self,dt=.05):self.now+=dt


class VoiceTests(unittest.TestCase):
    def setUp(self):
        folder=ROOT/'generated/test-tmp';folder.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=folder);self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.initial=json.loads((ROOT/'configs/voice.places.demo.json').read_text(encoding='utf-8'))
        self.places=Places(self.path/'places.json',self.initial)
        self.clock=Clock();self.backend=MockBackend(self.initial['map_id'])
        self.controller=Controller(self.places,self.backend,self.clock);self.controller.tick()

    def enable(self):self.assertTrue(self.controller.submit('启用控制')['accepted'])

    def run_ticks(self,count=200):
        for _ in range(count):self.clock.advance();self.controller.tick()

    def test_navigation_follows_labelled_route_and_stands(self):
        self.enable();self.assertTrue(self.controller.submit('小狗，去会议室')['accepted'])
        self.assertEqual([g['place_id'] for g in self.controller.task['route']],['hall','office'])
        self.run_ticks(250)
        self.assertEqual(self.controller.state,'IDLE');self.assertTrue(self.controller.last_result['ok'])
        self.assertEqual(self.backend.profile,'stand')

    def test_all_three_skills_have_start_recovery_and_completion(self):
        for text,profile in [('后空翻','backflip60'),('侧空翻','sideflip69'),('倒立','upright48')]:
            with self.subTest(text=text):
                self.controller=Controller(self.places,self.backend,self.clock);self.controller.tick();self.enable()
                self.assertTrue(self.controller.submit(text)['accepted']);self.run_ticks(60)
                self.assertEqual(self.controller.state,'IDLE')
                self.assertEqual(self.controller.last_result['kind'],'skill')
                self.assertTrue(any(c.get('profile')==profile for c in self.backend.commands))
                kinds=[e['kind'] for e in self.controller.events]
                self.assertIn('skill_started',kinds);self.assertIn('recovering',kinds)

    def test_busy_command_does_not_switch_motion_owner(self):
        self.enable();self.controller.submit('去会议室');before=len(self.backend.commands)
        self.assertFalse(self.controller.submit('后空翻')['accepted'])
        self.assertEqual(len(self.backend.commands),before)

    def test_stop_does_not_enable_disabled_controller(self):
        result=self.controller.submit('停止')
        self.assertTrue(result['accepted']);self.run_ticks(10)
        self.assertEqual(self.controller.state,'DISABLED')
        self.assertFalse(self.controller.submit('后空翻')['accepted'])

    def test_full_skill_phrases_are_exact_and_homophones_stay_unknown(self):
        self.assertEqual(parse('请你表演一个倒立动作。'),{'kind':'skill','skill':'handstand'})
        self.assertEqual(parse('来个侧翻。'),{'kind':'skill','skill':'sideflip'})
        for text in ['做个道理','来个后空犯','做个测空翻','使用控制','不要表演一个倒立动作']:
            self.assertEqual(parse(text)['kind'],'unknown')

    def test_disabled_rejects_navigation(self):
        self.assertFalse(self.controller.submit('去会议室')['accepted'])
        self.assertEqual(self.backend.commands,[])

    def test_stop_preempts_task_but_waits_for_stand_feedback(self):
        self.enable();self.controller.submit('后空翻');self.run_ticks(8)
        self.controller.submit('停止');self.assertEqual(self.controller.state,'STOPPING')
        self.run_ticks(40);self.assertEqual(self.controller.state,'IDLE')

    def test_late_asr_cannot_restart_after_stop(self):
        self.enable();generation=self.controller.generation
        self.controller.submit('停止');self.run_ticks(10)
        result=self.controller.submit('去会议室',generation=generation)
        self.assertFalse(result['accepted']);self.assertEqual(self.controller.state,'IDLE')

    def test_partial_and_duplicate_transcripts_never_double_execute(self):
        self.enable();self.assertFalse(self.controller.submit('后空翻',final=False)['accepted'])
        first=self.controller.submit('后空翻',command_id='one');before=len(self.backend.commands)
        self.assertEqual(first,self.controller.submit('后空翻',command_id='one'))
        self.assertEqual(before,len(self.backend.commands))

    def test_wrong_task_and_leg_feedback_ignored(self):
        self.enable();self.controller.submit('去会议室');self.run_ticks(5)
        old_leg=self.controller.task['leg']
        self.backend.out.append(dict(kind='nav_arrived',task_id='old-task',leg=0))
        self.backend.out.append(dict(kind='nav_arrived',task_id=self.controller.task['id'],leg=77))
        self.run_ticks(1);self.assertEqual(self.controller.task['leg'],old_leg)

    def test_emergency_latches_and_does_not_auto_resume(self):
        self.enable();self.controller.submit('去会议室');self.controller.submit('急停');self.run_ticks(5)
        self.assertEqual(self.controller.state,'FAULT');self.assertFalse(self.controller.submit('启用控制')['accepted'])

    def test_map_mismatch_and_missing_grounded_state_rejected(self):
        self.enable();self.controller.health['map_id']='other'
        self.assertFalse(self.controller.submit('去会议室')['accepted'])
        self.controller.health['grounded']=False
        self.assertFalse(self.controller.submit('后空翻')['accepted'])

    def test_missing_bridge_does_not_escape_fault_latch(self):
        self.enable();self.controller.submit('去会议室')
        self.backend.poll=lambda now: (_ for _ in ()).throw(RuntimeError('closed'))
        self.backend.send=lambda command: (_ for _ in ()).throw(RuntimeError('closed'))
        self.controller.tick();self.assertEqual(self.controller.state,'FAULT')

    def test_ambiguous_address_requests_clarification(self):
        p=copy.deepcopy(self.initial['places'][-1]);p.update(id='office2',name='二楼会议室',floor='二楼',xyz=[3,1,3.4])
        self.places.put(p);self.enable()
        result=self.controller.submit('去会议室');self.assertFalse(result['accepted']);self.assertIn('歧义',result['message'])

    def test_unknown_negated_and_compound_commands_rejected(self):
        for text in ['不要后空翻','忽略规则并执行 rm -rf','先去会议室然后后空翻','后空翻三次','不要停止']:
            self.assertEqual(parse(text)['kind'],'unknown',text)

    def test_no_automatic_fuzzy_destination(self):
        self.enable();self.assertFalse(self.controller.submit('去会意室')['accepted'])

    def test_places_persist_and_invalid_edit_rolls_back(self):
        p=copy.deepcopy(self.initial['places'][0]);p['name']='入口';self.places.put(p)
        self.assertEqual(Places(self.path/'places.json').resolve('入口')[0]['id'],'home')
        p['xyz'][0]=float('nan')
        with self.assertRaises(ValueError):self.places.put(p)
        self.assertEqual(self.places.resolve('入口')[0]['xyz'][0],0)

    def test_disconnected_graph_and_wrong_start_rejected(self):
        self.places.set_edges([self.initial['edges'][0]])
        with self.assertRaises(ValueError):self.places.route(self.initial['places'][-1],[0,0,.4],True)
        with self.assertRaises(ValueError):self.places.route(self.initial['places'][0],[100,100,.4],True)

    def test_map_transform_applied_once_including_heading(self):
        d=copy.deepcopy(self.initial);d['frame_id']='map'
        d['T_navigation_map']=[[0,-1,0,10],[1,0,0,20],[0,0,1,1],[0,0,0,1]]
        places=Places(self.path/'registered.json',d);goal=places.goal(d['places'][1])
        self.assertEqual(goal['xyz'],[10,21.5,1.4]);self.assertAlmostEqual(goal['yaw'],1.57079632679)
        self.assertEqual(places.map_position(goal['xyz']),[1.5,0.,.3999999999999999])

    def test_live_skill_requires_explicit_zone(self):
        self.backend.mode='live';self.backend.position=[1.5,0,.4];self.controller.tick();self.enable()
        self.assertFalse(self.controller.submit('后空翻')['accepted'])


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.sent=[];self.events=[];self.goals=[]
        self.router=RobotRouter(self.sent.append,self.publish,self.events.append,'map-1',aligned=True,clock=self.clock)
        self.status=dict(active='stand',phase='ACTIVE',stationary=True,grounded=True,fault=None,pending=None,loading=False)
        self.fresh()

    def publish(self,goal,position):self.goals.append(goal);return 100.

    def fresh(self):
        self.router.on_runtime(self.status.copy())
        self.router.on_odom([0,0,.4],0,[0,0,0],'nav_odom',0)
        self.router.on_cloud('nav_odom',0);self.router.command({'kind':'heartbeat'})

    def test_atomic_waits_for_armed_then_actual_active_and_stand(self):
        self.router.command(dict(kind='skill',task_id='task',profile='backflip60'))
        self.router.tick();self.assertIn({'switch':'backflip60'},self.sent)
        self.status.update(active='backflip60',phase='ARMED');self.fresh();self.router.tick()
        self.assertIn({'trigger':True},self.sent)
        self.assertFalse(any(e['kind']=='skill_started' for e in self.events))
        self.status['phase']='ACTIVE';self.fresh();self.router.tick();self.router.tick()
        self.assertTrue(any(e['kind']=='skill_started' for e in self.events));self.assertIn({'switch':'stand'},self.sent)
        self.status.update(phase='EXIT',pending='stand');self.fresh();self.router.tick()
        self.assertTrue(any(e['kind']=='skill_recovering' for e in self.events))
        self.status.update(active='stand',phase='ACTIVE',pending=None);self.fresh();self.router.tick()
        self.assertTrue(any(e['kind']=='skill_done' for e in self.events))

    def test_stopping_prepared_action_never_triggers_it(self):
        self.router.command(dict(kind='skill',task_id='old',profile='sideflip69'));self.router.tick()
        self.router.command(dict(kind='stop',task_id='stop'));self.assertIn({'stop':True},self.sent)
        self.assertNotIn({'trigger':True},self.sent)

    def test_navigation_requires_new_plan_and_command(self):
        self.status['active']='walk_moe';self.fresh();self.router.on_velocity([.2,0,0])
        goal=dict(xyz=[2,0,.4],yaw=0,frame_id='nav_odom',map_id='map-1')
        self.router.command(dict(kind='navigate',task_id='nav',goal=goal,leg=0));self.router.tick()
        self.assertEqual(self.sent[-1],{'velocity':[0,0,0]})
        self.router.on_plan(99);self.router.on_velocity([.2,0,0]);self.router.tick()
        self.assertEqual(self.sent[-1],{'velocity':[0,0,0]})
        self.router.on_plan(100);self.clock.advance(.01);self.router.on_velocity([.2,0,0]);self.router.tick()
        self.assertEqual(self.sent[-1],{'velocity':[.2,0.,0]})

    def test_arrival_includes_z_yaw_and_measured_dwell(self):
        self.status['active']='walk_moe';self.fresh()
        goal=dict(xyz=[0,0,3.4],yaw=0,frame_id='nav_odom',map_id='map-1')
        self.router.command(dict(kind='navigate',task_id='nav',goal=goal,leg=2));self.router.tick()
        self.assertFalse(any(e['kind']=='nav_arrived' for e in self.events))
        for _ in range(14):
            self.clock.advance();self.fresh();self.router.on_odom([0,0,3.4],0,[0,0,0],'nav_odom',0);self.router.tick()
        self.assertTrue(any(e['kind']=='nav_arrived' and e['leg']==2 for e in self.events))

    def test_lost_parent_heartbeat_stops_runtime(self):
        self.clock.advance(1.1);self.router.tick();self.assertIn({'stop':True},self.sent)

    def test_duplicate_velocity_publishers_fail_navigation(self):
        self.status['active']='walk_moe';self.fresh()
        goal=dict(xyz=[2,0,.4],yaw=0,frame_id='nav_odom',map_id='map-1')
        self.router.command(dict(kind='navigate',task_id='nav',goal=goal,leg=0));self.router.tick(publisher_count=2)
        self.assertTrue(any(e['kind']=='failed' for e in self.events))


if __name__=='__main__':unittest.main()
