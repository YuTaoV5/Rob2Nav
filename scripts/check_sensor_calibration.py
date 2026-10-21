"""Offline rigid-transform/holdout/FOV checks; no sensor, ROS, SDK or motor access."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def finite(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Required finite numeric measurement is missing')
    return float(value)

def vector(value,n):
    if not isinstance(value,list) or len(value)!=n:raise ValueError('Vector has wrong dimension')
    return [finite(x) for x in value]

def rigid(value):
    if not isinstance(value,list) or len(value)!=4:raise ValueError('Measured 4x4 transform required')
    m=[vector(row,4) for row in value]
    if max(abs(m[3][i]-[0,0,0,1][i]) for i in range(4))>1e-8:raise ValueError('Invalid homogeneous row')
    r=[row[:3] for row in m[:3]]
    err=max(abs(sum(r[k][i]*r[k][j] for k in range(3))-int(i==j)) for i in range(3) for j in range(3))
    det=(r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if err>1e-5 or abs(det-1)>1e-5:raise ValueError('Rotation must be orthonormal with determinant +1')
    return m

def transform(m,p):return [sum(m[i][j]*p[j] for j in range(3))+m[i][3] for i in range(3)]
def inverse_point(m,p):return [sum(m[j][i]*(p[j]-m[j][3]) for j in range(3)) for i in range(3)]
def percentile(xs,q):
    ys=sorted(xs);k=(len(ys)-1)*q;i=int(k);return ys[i]+(ys[min(i+1,len(ys)-1)]-ys[i])*(k-i)

def check(data,allow_synthetic=False):
    origin=data.get('data_origin')
    if origin!='measured' and not (allow_synthetic and origin=='synthetic_example'):
        raise ValueError('Unmeasured template or synthetic input refused; use real measurements, or explicitly --allow-synthetic for the example')
    if not all(data.get(k) for k in ('calibration_id','camera_serial','base_frame_id','camera_frame_id')):
        raise ValueError('Calibration ID, serial and frame IDs required')
    m=rigid(data.get('T_base_color_optical'));checked=['T_base_color_optical']
    for key in ('T_base_lidar','T_imu_lidar'):
        if data.get(key) is not None:rigid(data[key]);checked.append(key)
    camera=data['camera']
    if camera.get('projection_model')!='rectified_pinhole':raise ValueError('Tool requires rectified pinhole intrinsics; raw distorted images need the matching SDK/model')
    w,h,fx,fy,cx,cy,zmin,zmax=[finite(camera.get(k)) for k in ('width','height','fx','fy','cx','cy','minimum_depth_m','maximum_depth_m')]
    if min(w,h,fx,fy,zmin)<=0 or zmax<=zmin or not(0<=cx<w and 0<=cy<h):raise ValueError('Invalid camera geometry/range')
    limits=data['limits'];errors=[]
    pairs=data.get('holdout_pairs',[])
    if finite(limits['minimum_holdout_points'])<6:raise ValueError('At least six independent holdout points required')
    if len(pairs)<finite(limits['minimum_holdout_points']):raise ValueError('Independent holdout point pairs required')
    for pair in pairs:
        a=transform(m,vector(pair['camera_point_m'],3));b=vector(pair['base_reference_m'],3)
        errors.append(math.sqrt(sum((a[i]-b[i])**2 for i in range(3))))
    p95=percentile(errors,.95);maximum=max(errors)
    pixels=[];margin=finite(limits['image_margin_px'])
    if margin<0 or 2*margin>=min(w,h):raise ValueError('Invalid image margin')
    for point in data.get('workspace_points_base',[]):
        c=inverse_point(m,vector(point,3));z=c[2]
        u,v=(fx*c[0]/z+cx,fy*c[1]/z+cy) if z>0 else (None,None)
        inside=zmin<=z<=zmax and margin<=u<w-margin and margin<=v<h-margin
        pixels.append(dict(point_base_m=point,optical_z_m=z,pixel=[u,v],inside_geometric_view=inside))
    if not pixels:raise ValueError('Workspace sample points required')
    age=finite(data['timing']['measurement_age_p99_ms']);skew=finite(data['timing']['rgb_depth_skew_p99_ms'])
    if min(age,skew)<0:raise ValueError('Timing measurements cannot be negative')
    if min(finite(limits[k]) for k in ('position_p95_m','position_max_m','measurement_age_p99_ms','rgb_depth_skew_p99_ms'))<=0:
        raise ValueError('Positive error/time limits required')
    passed=(p95<=limits['position_p95_m'] and maximum<=limits['position_max_m'] and all(p['inside_geometric_view'] for p in pixels)
            and age<=limits['measurement_age_p99_ms'] and skew<=limits['rgb_depth_skew_p99_ms'])
    return dict(offline_checks_passed=passed,hardware_release=False,data_origin=origin,calibration_id=data['calibration_id'],
                checked_transforms=checked,holdout_count=len(errors),position_p95_m=p95,position_max_m=maximum,
                measurement_age_p99_ms=age,rgb_depth_skew_p99_ms=skew,workspace_projections=pixels,
                scope='Offline geometry and supplied measurement checks only; no calibration solving, sensor precision, occlusion, contact or hardware qualification')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-synthetic',action='store_true');a=p.parse_args()
    raw=a.input.read_bytes()
    try:report=check(json.loads(raw.decode('utf-8-sig')),a.allow_synthetic)
    except (ValueError,KeyError,TypeError) as e:report=dict(offline_checks_passed=False,hardware_release=False,error=str(e))
    report['input_sha256']=hashlib.sha256(raw).hexdigest();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='workspace_projections'},ensure_ascii=False))
    raise SystemExit(0 if report['offline_checks_passed'] else 2)
if __name__=='__main__':main()
