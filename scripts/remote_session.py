"""Interactive SSH/SFTP helper; password is prompted and never saved.

Host keys must already be in the user's OpenSSH known_hosts. JSON lines accept
run (command), put/get (local, remote), and exit. Intended for reproducible runs.
"""
import getpass
import argparse
import json
from pathlib import Path
import sys
import paramiko
import socketserver
import select
import threading

parser = argparse.ArgumentParser()
parser.add_argument("host")
parser.add_argument("--port", type=int, default=22)
parser.add_argument("--user", default="root")
args = parser.parse_args()
client = paramiko.SSHClient()
known = Path.home()/".ssh/known_hosts"
client.load_host_keys(str(known))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(args.host, port=args.port, username=args.user, password=getpass.getpass("SSH password: "), timeout=20)
client.get_transport().set_keepalive(30)
print("CONNECTED", flush=True)
for line in sys.stdin:
    try:
        task = json.loads(line)
        if task["op"] == "exit":
            break
        if task['op']=='tunnel':
            transport=client.get_transport();remote_port=int(task.get('remote_port',8765))
            class Forward(socketserver.BaseRequestHandler):
                def handle(self):
                    channel=None
                    try:
                        channel=transport.open_channel('direct-tcpip',('127.0.0.1',remote_port),self.request.getpeername())
                        if channel is None:return
                        while True:
                            ready,_,_=select.select([self.request,channel],[],[],30)
                            for source in ready:
                                data=source.recv(65536)
                                if not data:return
                                (channel if source is self.request else self.request).sendall(data)
                    except (OSError,EOFError,paramiko.SSHException):pass
                    finally:
                        if channel is not None:channel.close()
            class Server(socketserver.ThreadingTCPServer):
                allow_reuse_address=True
                daemon_threads=True
            server=Server(('127.0.0.1',int(task.get('local_port',8766))),Forward)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            print('TUNNEL READY',flush=True)
            continue
        if task["op"] == "run":
            _, stdout, stderr = client.exec_command(task["command"])
            print(stdout.read().decode(), end="", flush=True)
            print(stderr.read().decode(), end="", flush=True)
            print("EXIT", stdout.channel.recv_exit_status(), flush=True)
        else:
            with client.open_sftp() as sftp:
                if task["op"] == "put":
                    sftp.put(task["local"], task["remote"])
                elif task["op"] == "get":
                    sftp.get(task["remote"], task["local"])
                else:
                    raise ValueError("Unknown operation")
            print("TRANSFERRED", flush=True)
    except Exception as error:
        print(type(error).__name__, str(error), flush=True)
client.close()
