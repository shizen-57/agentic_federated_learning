"""Fingerprint of the simulator's source code, saved with every run.

The sweep uses it to tell a finished run made by the current code from one
left over from an older version, which must be re-run rather than reused.
"""
from __future__ import annotations

import hashlib
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def code_hash() -> str:
    h = hashlib.sha1()
    for root, _, files in sorted(os.walk(_HERE)):
        if "__pycache__" in root:
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                path = os.path.join(root, f)
                h.update(os.path.relpath(path, _HERE).encode())
                with open(path, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()[:16]
