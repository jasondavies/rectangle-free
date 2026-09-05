#!/usr/bin/env python3
"""Content identity of the partition solver, including its build recipe."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = [root / 'Makefile', Path(__file__).resolve()]
paths += sorted((root / 'src/partition').glob('*.[ch]'))
paths += [root / 'src/common/sha256_c.c', root / 'src/common/sha256_c.h']
h = hashlib.sha256()
for path in paths:
    h.update(str(path.relative_to(root)).encode() + b'\0')
    data = path.read_bytes()
    h.update(str(len(data)).encode() + b'\0' + data)
print(h.hexdigest())
