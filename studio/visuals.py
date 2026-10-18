"""Read-only render assets, derived from the actual model and terrain generators."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROBOT_DIR = Path('vendor/RoboGauge/resources/robots/go2')


def go2_urdf(root):
    """Kinematic display conversion, not a dynamics/export replacement for MJCF."""
    source = ET.parse(root / ROBOT_DIR / 'go2.xml').getroot()
    robot = ET.Element('robot', name='RoboGauge_Go2_display')
    defaults = {}
    def collect(node, inherited):
        data = {k: dict(v) for k, v in inherited.items()}
        for child in node:
            if child.tag != 'default': data.setdefault(child.tag, {}).update(child.attrib)
        if node.get('class'): defaults[node.get('class')] = data
        for child in node.findall('default'): collect(child, data)
    collect(source.find('default'), {})
    materials = {m.get('name'): m.get('rgba') for m in source.findall('asset/material')}
    meshes = {m.get('name', Path(m.get('file')).stem): m.get('file') for m in source.findall('asset/mesh')}
    def origin(node):
        xyz = node.get('pos', '0 0 0'); rpy = '0 0 0'
        if node.get('quat'):
            w,x,y,z = map(float,node.get('quat').split()); n=math.sqrt(w*w+x*x+y*y+z*z)
            w,x,y,z = (a/n for a in (w,x,y,z))
            rpy = ' '.join(map(str,(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))))
        return {'xyz':xyz,'rpy':rpy}
    def body(b, parent=None):
        name=b.get('name'); link=ET.SubElement(robot,'link',name=name)
        if parent:
            j=b.find('joint'); attrs=defaults.get(j.get('class'),{}).get('joint',{}).copy() if j is not None else {}
            if j is not None: attrs.update(j.attrib)
            joint=ET.SubElement(robot,'joint',name=attrs.get('name',name+'_fixed'),type='revolute' if j is not None else 'fixed')
            ET.SubElement(joint,'parent',link=parent);ET.SubElement(joint,'child',link=name);ET.SubElement(joint,'origin',**origin(b))
            if j is not None:
                lo,hi=attrs.get('range','-3.14 3.14').split()
                ET.SubElement(joint,'axis',xyz=attrs.get('axis','0 1 0'))
                ET.SubElement(joint,'limit',lower=lo,upper=hi,effort='45.43',velocity='30')
        for geom in b.findall('geom'):
            attrs=defaults.get(geom.get('class'),{}).get('geom',{}).copy();attrs.update(geom.attrib)
            mode='visual' if attrs.get('mesh') else 'collision'
            out=ET.SubElement(link,mode);ET.SubElement(out,'origin',**origin(ET.Element('geom',attrs)))
            geometry=ET.SubElement(out,'geometry');size=list(map(float,attrs.get('size','0.02').split()));typ=attrs.get('type','sphere')
            if attrs.get('mesh'): ET.SubElement(geometry,'mesh',filename='/api/robot/assets/'+meshes[attrs['mesh']])
            elif typ=='box': ET.SubElement(geometry,'box',size=' '.join(str(v*2) for v in size))
            elif typ=='cylinder': ET.SubElement(geometry,'cylinder',radius=str(size[0]),length=str(size[1]*2))
            else: ET.SubElement(geometry,'sphere',radius=str(size[0]))
            material=ET.SubElement(out,'material',name=attrs.get('material','collision'))
            ET.SubElement(material,'color',rgba=materials.get(attrs.get('material'),'0.2 0.5 1 0.35'))
        for child in b.findall('body'): body(child,name)
    body(source.find('worldbody/body'))
    return ET.tostring(robot,encoding='unicode')


def terrain_data(kind, course, difficulty):
    from perceptive_rl.geometry import lane_boxes
    from perceptive_rl import ame_terrain
    from moe_perceptive import sparse_course
    if kind not in ('flat','stairs','platform','ame_stakes','ame_alternate'): raise ValueError(kind)
    if course not in ('original','wide_deep') or not 0 <= difficulty <= 1: raise ValueError(course)
    sparse=kind.startswith('ame_'); generator=sparse_course if course=='wide_deep' else ame_terrain
    boxes=generator.boxes(difficulty,kind) if sparse else lane_boxes(difficulty,kind,(8.,8.))
    return {'kind':kind,'course':course,'difficulty':difficulty,'boxes':boxes,
            'parameters':generator.parameters(difficulty,kind) if sparse else {},
            'size':[8,8], 'source':'moe_perceptive/sparse_course.py' if sparse and course=='wide_deep' else 'perceptive_rl/ame_terrain.py' if sparse else 'perceptive_rl/geometry.py',
            'note':'几何参考图；不是仿真结果。original 训练中可能启用浅坑课程，此图展示严格原几何；预览难度不改写训练课程。'}


def terrain_svg(data):
    w,h=data['size'];sx=560/w;sy=280/h
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 340" role="img"><rect width="600" height="340" rx="12" fill="#f0f5ff"/>']
    for x0,x1,y0,y1,top,bottom in sorted(data['boxes'],key=lambda b:b[4]):
        color='#203657' if top < -1 else '#719ef3' if top>0.01 else '#a7c5fa'
        out.append(f'<rect x="{20+x0*sx:.2f}" y="{20+(h-y1)*sy:.2f}" width="{(x1-x0)*sx:.2f}" height="{(y1-y0)*sy:.2f}" fill="{color}" stroke="#f8fbff" stroke-width=".4"/>')
    out.append(f'<text x="20" y="325" font-family="sans-serif" font-size="12" fill="#3b5b87">{w} m × {h} m · TOP VIEW · difficulty {data["difficulty"]:.2f}</text></svg>')
    return ''.join(out)
