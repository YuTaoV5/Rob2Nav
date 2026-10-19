"""CLI remains usable without FastAPI or a browser."""
import argparse,json
from pathlib import Path
from .contracts import profiles


def main():
    parser=argparse.ArgumentParser(prog='rob2nav')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('profiles',help='Export reviewed policy contracts')
    sim=commands.add_parser('simulate',help='Run a pinned source policy in MuJoCo, simulation only')
    sim.add_argument('--policy',choices=['pie','lsd_jump','lsd_backflip','lsd_sideflip'],required=True)
    sim.add_argument('--scene',choices=['empty','stairs'],default='empty')
    sim.add_argument('--seconds',type=float,default=6)
    sim.add_argument('--seed',type=int,default=801)
    sim.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    sim.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='profiles':print(json.dumps(profiles(),indent=2));return
    if not 0<args.seconds<=120:parser.error('Simulation duration must be in (0,120] seconds')
    if args.policy!='pie' and args.scene!='empty':parser.error('Motion references currently require flat terrain')
    from .backends.source_mujoco import pie,lsd
    if args.policy=='pie':pie(args.root,args.output,args.scene,args.seconds,args.seed)
    else:lsd(args.root,args.output,args.policy,args.seconds,args.seed)


if __name__=='__main__':main()
