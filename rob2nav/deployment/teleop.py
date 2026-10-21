"""Linux terminal keyboard client for the qualified, single-owner device service."""
import json
import select
import subprocess
import sys
import termios
import time
import tty


KEYS={
    'w':(.18,0.,0.), 's':(-.12,0.,0.),
    'a':(0.,.10,0.), 'd':(0.,-.10,0.),
    'q':(0.,0.,.35), 'e':(0.,0.,-.35),
    ' ':(0.,0.,0.),
}


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
        print('W/S forward/back, A/D lateral, Q/E yaw; SPACE stand, ESC stop. Hold key for motion.',flush=True)
        velocity=(0.,0.,0.);last_key=0.;last_send=0.
        while proc.poll() is None:
            readable,_,_=select.select([sys.stdin],[],[],.04)
            if readable:
                key=sys.stdin.read(1).lower()
                if key=='\x1b':break
                if key in KEYS:velocity=KEYS[key];last_key=time.monotonic()
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
