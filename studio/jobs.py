"""Durable queue shared by API and the independent, single-slot worker."""
import json
import sqlite3
import time
import uuid
from pathlib import Path


class Jobs:
    def __init__(self,root):
        self.directory=Path(root)/'outputs/studio'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.database=self.directory/'studio.sqlite3'
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, name TEXT, state TEXT, config TEXT, created REAL, updated REAL, detail TEXT)')

    def connect(self):
        db=sqlite3.connect(self.database,timeout=10);db.row_factory=sqlite3.Row
        return db

    def all(self):
        with self.connect() as db:
            return [self.unpack(v) for v in db.execute('SELECT * FROM jobs ORDER BY created DESC')]

    def unpack(self,row):
        d=dict(row);d['config']=json.loads(d['config']);d['detail']=json.loads(d['detail'] or '{}')
        return d

    def get(self,key):
        with self.connect() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(key,)).fetchone()
        if row is None:raise KeyError(key)
        return self.unpack(row)

    def create(self,config):
        key=uuid.uuid4().hex[:16];now=time.time()
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)',(key,config['name'],'draft',json.dumps(config),now,now,'{}'))
        return self.get(key)

    def update(self,key,state,detail=None):
        with self.connect() as db:
            db.execute('UPDATE jobs SET state=?, updated=?, detail=? WHERE id=?',(state,time.time(),json.dumps(detail or {}),key))

    def transition(self,key,old,new):
        with self.connect() as db:
            cur=db.execute('UPDATE jobs SET state=?,updated=? WHERE id=? AND state=?',(new,time.time(),key,old))
        return cur.rowcount==1

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            # An unknown former job may still own a GPU child. Never launch a
            # new task until the operator has reconciled that process.
            if db.execute("SELECT 1 FROM jobs WHERE state='unknown' LIMIT 1").fetchone():return None
            row=db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET state='preflight',updated=? WHERE id=?",(time.time(),row['id']))
        return self.unpack(row) if row else None

    def folder(self,key):
        # IDs must have come from the database, never accept arbitrary paths.
        self.get(key)
        return self.directory/'jobs'/key
