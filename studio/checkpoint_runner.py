"""Execute exactly one inference job, never claim or resume queued training."""
import argparse
import os
import signal
import time
from pathlib import Path
from .jobs import Jobs
from .worker import execute

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--python',required=True);p.add_argument('--lock-fd',type=int,required=True);a=p.parse_args()
    # Parent hands the already-held GPU flock to this child. Never unlock it
    # before its simulation subprocess has ended.
    os.fstat(a.lock_fd);jobs=Jobs(a.root);job=jobs.get(a.job);origin=dict(job['detail'])
    if job['config'].get('task')!='simulation' or job['config'].get('simulation_policy')!='checkpoint':
        raise ValueError('Inference runner refuses non-checkpoint or training tasks')
    def cancel(*_):
        (jobs.folder(a.job)/'cancel.flag').touch()
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        execute(jobs,job,a.root,a.python)
    except InterruptedError as e:jobs.update(a.job,'canceled',dict(reason=str(e)))
    except Exception as e:jobs.update(a.job,'failed',dict(reason=str(e)))
    finally:
        current=jobs.get(a.job);jobs.update(a.job,current['state'],dict(origin,**current['detail']))
        os.close(a.lock_fd)
if __name__=='__main__':main()
