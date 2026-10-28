#!/usr/bin/env python3
"""Create an EMPTY annotation registry bound to an actual PCD and coordinate frame."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_agent.places import Places
from voice_agent.pcd import file_sha256, projection


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pcd',type=Path,required=True);p.add_argument('--places',type=Path,required=True)
    p.add_argument('--frame',required=True);p.add_argument('--navigation-frame',default='nav_odom')
    p.add_argument('--transform',type=Path,help='JSON 4x4 T_navigation_map; measured map registration, not mounting angle')
    args=p.parse_args()
    if args.places.exists():p.error('标注文件已存在，请选新文件以保留历史')
    projection(args.pcd,max_points=20)
    digest=file_sha256(args.pcd)
    data=dict(schema=1,map_id='map-'+digest[:16],pcd_sha256=digest,frame_id=args.frame,
        navigation_frame=args.navigation_frame,T_navigation_map=json.loads(args.transform.read_text(encoding='utf-8')) if args.transform else None,
        route_start_radius=1.,places=[],edges=[])
    places=Places(args.places,data);places.save()
    print(json.dumps(dict(places=str(args.places),map_id=data['map_id'],points_frame=args.frame),ensure_ascii=False))


if __name__=='__main__':main()
