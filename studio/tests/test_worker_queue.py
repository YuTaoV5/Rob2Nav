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
