from studio.worker import drain_queue


def test_long_training_does_not_consume_idle_budget():
    now=[0.]
    pending=[{'id':'first'},{'id':'second'}]
    completed=[]
    class Queue:
        def claim(self):return pending.pop(0) if pending else None
        def update(self,*args):raise AssertionError(args)
    def run(job):
        now[0]+=3600
        completed.append(job['id'])
    def sleep(dt):now[0]+=dt
    drain_queue(Queue(),run,clock=lambda:now[0],sleep=sleep,idle_seconds=60)
    assert completed==['first','second']
    assert now[0]==7260


def test_failed_long_job_does_not_strand_next_job():
    now=[0.];pending=[{'id':'bad'},{'id':'good'}];updates=[];completed=[]
    class Queue:
        def claim(self):return pending.pop(0) if pending else None
        def update(self,*args):updates.append(args)
    def run(job):
        now[0]+=90
        if job['id']=='bad':raise RuntimeError('failed training')
        completed.append(job['id'])
    def sleep(dt):now[0]+=dt
    drain_queue(Queue(),run,clock=lambda:now[0],sleep=sleep,idle_seconds=2)
    assert completed==['good']
    assert updates==[('bad','failed',{'reason':'failed training'})]


def test_parent_dispatches_claimed_job_to_fresh_module_and_checks_terminal(tmp_path,monkeypatch):
    import studio.worker as worker
    import pytest
    state=['canceled'];calls=[]
    class Queue:
        def get(self,key):return {'state':state[0]}
    def run(cmd,**kwargs):calls.append((cmd,kwargs))
    monkeypatch.setattr(worker.subprocess,'run',run)
    worker.run_isolated(Queue(),{'id':'claimed123'},tmp_path,'/isaac/python.sh','/cpu/python')
    cmd,kwargs=calls[0]
    assert cmd[1:3]==['-m','studio.worker']
    assert cmd[cmd.index('--job')+1]=='claimed123'
    assert kwargs['cwd']==tmp_path and kwargs['check'] is True
    state[0]='running'
    with pytest.raises(RuntimeError,match='without a terminal'):worker.run_isolated(Queue(),{'id':'claimed123'},tmp_path,'/isaac/python.sh','/cpu/python')
