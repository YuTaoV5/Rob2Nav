"""Linux terminal keyboard client for the qualified, single-owner device service."""
import json
import select
import subprocess
import sys
import termios
import time
import tty
from .keys import VELOCITY_KEYS,key_request


def main():
    if not sys.stdin.isatty():raise SystemExit('A local Linux terminal is required for keyboard control')
    args=sys.argv[1:]
    if args[:1]==['--']:args=args[1:]
    if not args or args[0]!='run':raise SystemExit('Usage: python -m rob2nav.deployment.teleop -- run [qualified device service arguments]')
    command=[sys.executable,'-m','rob2nav.deployment',*args]
    old=termios.tcgetattr(sys.stdin.fileno());proc=None
    try:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,text=True,bufsize=1)
        tty.setcbreak(sys.stdin.fileno())
        print('W/S forward/back, A/D lateral, Q/E yaw; SPACE zero velocity; 1 stand, 2 walk, 3 sideflip, 4 backflip, 5 upright; T trigger armed action; ESC stop.',flush=True)
        velocity=(0.,0.,0.);last_key=0.;last_send=0.
        while proc.poll() is None:
            readable,_,_=select.select([sys.stdin],[],[],.04)
            if readable:
                key=sys.stdin.read(1).lower()
                request=key_request(key)
                if request=={'stop':True}:break
                if key in VELOCITY_KEYS:velocity=VELOCITY_KEYS[key];last_key=time.monotonic()
                elif request is not None:
                    try:proc.stdin.write(json.dumps(request)+'\n');proc.stdin.flush()
                    except BrokenPipeError:break
            now=time.monotonic()
            if now-last_key>.18:velocity=(0.,0.,0.)
            if now-last_send>=.08:
                try:proc.stdin.write(json.dumps({'velocity':velocity})+'\n');proc.stdin.flush()
                except BrokenPipeError:break
                last_send=now
    finally:
        termios.tcsetattr(sys.stdin.fileno(),termios.TCSADRAIN,old)
        if proc is not None and proc.poll() is None:
            try:proc.stdin.write('{"stop":true}\n');proc.stdin.flush()
            except (BrokenPipeError,OSError):pass
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=3)


if __name__=='__main__':main()
