import unittest
import xml.etree.ElementTree as ET
import numpy as np
from skills.navigation.tasks import populate,StaticRoute,STEP_TASK_SCENES


def geometry(scene):
    world=ET.Element('worldbody');names=[];meta=populate(world,scene,{},names)
    boxes=[]
    for g in world.findall('geom'):
        if g.get('name') not in names:continue
        x,y,_=map(float,g.get('pos').split());sx,sy,_=map(float,g.get('size').split())
        boxes.append(dict(x=x,y=y,sx=sx,sy=sy))
    return world,names,meta,boxes


class ScanStepRoutesTests(unittest.TestCase):
    def test_steps_are_physical_but_not_blocking_floorplan_boxes(self):
        for scene in STEP_TASK_SCENES:
            world,names,meta,boxes=geometry(scene);router=StaticRoute(boxes)
            self.assertTrue(all(world.find(f"geom[@name='{name}']") is not None for name in meta['step_geom_names']))
            self.assertFalse(set(names)&set(meta['step_geom_names']))
            self.assertTrue(router.free([2.7,0]))
            self.assertTrue(router.path([0,0],meta['goal']))

    def test_door_route_crosses_opening_and_overhead_does_not_seal_map(self):
        world,names,meta,boxes=geometry('stairs_door');router=StaticRoute(boxes)
        self.assertNotIn('door_lintel',names);self.assertIsNotNone(world.find("geom[@name='door_lintel']"))
        path=router.path([0,0],meta['goal']);self.assertGreater(len(path),1)
        crossings=[]
        for a,b in zip(path,path[1:]):
            self.assertTrue(router.line(a,b))
            if a[0]<=meta['door']['x']<=b[0]:
                t=(meta['door']['x']-a[0])/max(b[0]-a[0],1e-9)
                crossings.append(a[1]+t*(b[1]-a[1]))
        self.assertTrue(crossings)
        self.assertTrue(all(abs(y)<=meta['door']['width_m']/2-.37 for y in crossings))
        self.assertFalse(router.free([meta['door']['x'],.8]))

    def test_course_dimensions_and_return_to_floor(self):
        for scene,height in [('stairs_door',.05),('stairs_route',.08)]:
            _,_,m,_=geometry(scene)
            self.assertEqual(m['step_height_m'],height)
            self.assertAlmostEqual(max(s[2] for s in m['surfaces']),2*height)
            self.assertGreater(m['goal'][0],m['surfaces'][-1][1]+.5)

    def test_free_pose_rounding_inside_furniture_does_not_deadlock_astar(self):
        _,_,m,boxes=geometry('stairs_route');router=StaticRoute(boxes)
        start=[3.313621780748822,.3139925636446596]
        self.assertTrue(router.free(start))
        self.assertFalse(router.free(np.round(np.array(start)/router.res)*router.res))
        path=router.path(start,m['goal'])
        self.assertGreater(len(path),1)
        self.assertTrue(all(router.line(a,b) for a,b in zip(path,path[1:])))

    def test_larger_step_route_clearance_still_connects_both_sides(self):
        _,_,m,boxes=geometry('stairs_route');router=StaticRoute(boxes,clearance=.60)
        path=router.path([0,0],m['goal']);self.assertGreater(len(path),1)
        self.assertTrue(all(router.line(a,b) for a,b in zip(path,path[1:])))


if __name__=='__main__':unittest.main()
