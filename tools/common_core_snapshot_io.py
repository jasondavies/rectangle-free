"""Format-neutral SQLite backup/publication for new campaign controllers.

The caller validates provenance and ownership through `check(path)`. The live
v2 snapshot entry point is intentionally unchanged until its campaign ends.
"""
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import common_core_campaign as cc


def snapshot(source, destination, check, timeout=60.):
    cc.require(math.isfinite(timeout) and timeout > 0, 'positive backup timeout required')
    source, destination = Path(source).absolute(), Path(destination).absolute()
    cc.require(source != destination and (not destination.exists() or not os.path.samefile(source,destination)),
               'snapshot destination must differ from source')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with cc.claim(destination):
        fd,name=tempfile.mkstemp(prefix='.'+destination.name+'.',dir=destination.parent)
        os.close(fd); temporary=Path(name); started=time.monotonic()
        try:
            reader=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=30.)
            try:
                writer=sqlite3.connect(temporary)
                try:
                    def progress(status,remaining,total):
                        cc.require(time.monotonic()-started < timeout,'live backup timeout')
                    reader.backup(writer,pages=256,sleep=.01,progress=progress)
                finally:writer.close()
            finally:reader.close()
            report=check(temporary)
            if destination.exists():
                previous=check(destination)
                cc.require(previous['identity'] == report['identity'],'snapshot identity changed')
                db=sqlite3.connect(destination.as_uri()+'?mode=ro',uri=True)
                try:
                    db.execute('ATTACH DATABASE ? AS candidate',(temporary.as_uri()+'?mode=ro',))
                    for table,columns in [('groups','gid,hash'),('ranges','gid,pi,begin,hash')]:
                        missing=db.execute(f'SELECT 1 FROM (SELECT {columns} FROM main.{table} '
                                           f'EXCEPT SELECT {columns} FROM candidate.{table}) LIMIT 1').fetchone()
                        cc.require(missing is None,'snapshot would discard/change committed records')
                finally:db.close()
            checksum=cc.file_digest(temporary)
            with temporary.open('rb') as f:os.fsync(f.fileno())
            os.replace(temporary,destination)
            directory=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
            return dict(status='snapshot_published',sha256=checksum,path=str(destination),
                        groups=report['groups'],ranges=report['ranges'],seconds=time.monotonic()-started)
        finally:
            if temporary.exists():temporary.unlink()
