#!/usr/bin/env python3
"""Publish a verified, closed SQLite snapshot; never copy a live DB with scp.

Copy the published file off-host after this command completes. A local snapshot
is not an off-host backup. Atomic publication keeps a previous good copy intact
on failure. Both publication and the containing directory are fsynced.
"""
import argparse
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import common_core_campaign as cc


def verify_snapshot(path, expected_sha256):
    """Verify a closed downloaded snapshot against its publisher's receipt."""
    path = Path(path)
    cc.require(len(expected_sha256) == 64 and all(c in '0123456789abcdef' for c in expected_sha256),
               'invalid expected SHA-256')
    cc.require(cc.file_digest(path) == expected_sha256, 'downloaded snapshot checksum mismatch')
    journal = cc.Journal(path)
    try:
        cc.require(journal.identity['format'] == cc.FORMAT, 'not a production journal')
        groups = ranges = 0
        for _, _, records in journal.ordered_groups():
            groups += 1
            ranges += sum(len(r) for r in records)
    finally:
        journal.close()
    cc.require(cc.file_digest(path) == expected_sha256, 'snapshot changed during verification')
    return dict(status='download_verified',sha256=expected_sha256,groups=groups,ranges=ranges)


def snapshot(source, destination, backup_timeout=60.):
    """One publisher per destination; fail rather than race another snapshot."""
    cc.require(math.isfinite(backup_timeout) and backup_timeout > 0, 'positive backup timeout required')
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with cc.claim(destination):
        return _snapshot(source, destination, backup_timeout)


def _snapshot(source, destination, backup_timeout):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    cc.require(source != destination and (not destination.exists() or not os.path.samefile(source,destination)),
               'snapshot destination must differ from source')
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+destination.name+'.',dir=destination.parent)
    os.close(fd)
    temporary=Path(name)
    started=time.monotonic()
    reader=writer=check=previous=None
    try:
        reader=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=30.)
        writer=sqlite3.connect(temporary)
        def progress(status, remaining, total):
            cc.require(time.monotonic()-started < backup_timeout, 'live backup timeout')
        reader.backup(writer,pages=256,sleep=.01,progress=progress)
        writer.close();writer=None
        reader.close();reader=None  # release source locks before validation
        check=cc.Journal(temporary)
        cc.require(check.identity['format']==cc.FORMAT,'not a production journal')
        groups=ranges=0
        for _,_,records in check.ordered_groups():
            groups+=1;ranges+=sum(len(r) for r in records)
        identity=check.identity
        check.close();check=None
        if destination.exists():
            previous=cc.Journal(destination)
            cc.require(previous.identity==identity,'snapshot identity changed')
            previous.db.execute('ATTACH DATABASE ? AS candidate',(temporary.as_uri()+'?mode=ro',))
            for table,columns in [('groups','gid,hash'),('ranges','gid,pi,begin,hash')]:
                missing=previous.db.execute(f'SELECT 1 FROM (SELECT {columns} FROM main.{table} '
                            f'EXCEPT SELECT {columns} FROM candidate.{table}) LIMIT 1').fetchone()
                cc.require(missing is None,'snapshot would discard/change committed records')
            previous.close();previous=None
        checksum=cc.file_digest(temporary)
        with temporary.open('rb') as file:os.fsync(file.fileno())
        os.replace(temporary,destination)
        directory=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        return dict(status='snapshot_published',path=str(destination),sha256=checksum,
                    groups=groups,ranges=ranges,seconds=time.monotonic()-started)
    finally:
        for connection in (reader,writer):
            if connection:connection.close()
        for journal in (check,previous):
            if journal:journal.close()
        if temporary.exists():temporary.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path)
    mode.add_argument('--verify-sha256',help='verify a closed download against the publication receipt')
    p.add_argument('--backup-timeout',type=float,default=60.,help='bound the live SQLite copy, in seconds')
    a=p.parse_args()
    result=(verify_snapshot(a.source,a.verify_sha256) if a.verify_sha256 else
            snapshot(a.source,a.output,a.backup_timeout))
    print(cc.encoded(result),flush=True)


if __name__=='__main__':main()
