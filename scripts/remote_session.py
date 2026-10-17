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
print("CONNECTED", flush=True)
for line in sys.stdin:
    try:
        task = json.loads(line)
        if task["op"] == "exit":
            break
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
