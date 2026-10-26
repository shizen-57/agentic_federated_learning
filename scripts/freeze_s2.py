#!/usr/bin/env python3
"""Re-freeze the s2 protocol: write configs/s2_frozen.json and pin it.

Records the simulator fingerprint, the SHA-256 of every experiment input
(prespec_s2.MANIFEST_INPUTS) and the test definitions, then writes the code
fingerprint and the manifest's own SHA-256 into scripts/prespec_s2.py, so the
two pins are always updated together.

Use it only when deliberately revising the protocol, after the code and
configs are final and reviewed, and never to admit results that were made
with a different version after looking at them (docs/ADVANCED_RESULTS.md).

    python scripts/freeze_s2.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prespec_s2 as P  # noqa: E402
from afl.provenance import code_hash  # noqa: E402


def main() -> int:
    code = code_hash()
    inputs = {}
    for path in P.MANIFEST_INPUTS:
        with open(os.path.join(P.ROOT, path), "rb") as fh:
            inputs[path] = hashlib.sha256(fh.read()).hexdigest()
    manifest = {
        "schema_version": 1,
        "code_hash": code,
        "input_sha256": inputs,
        "protocol": json.loads(json.dumps({"seeds": sorted(P.SEEDS), "families": P.FAMILIES,
                                           "higher_is_better": P.HIGHER_IS_BETTER})),
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with open(P.MANIFEST_PATH, "wb") as fh:
        fh.write(data)
    sha = hashlib.sha256(data).hexdigest()
    path = os.path.join(HERE, "prespec_s2.py")
    with open(path) as fh:
        src = fh.read()
    src, n1 = re.subn(r'FROZEN_CODE_HASH = "[0-9a-f]*"[^\n]*', f'FROZEN_CODE_HASH = "{code}"', src)
    src, n2 = re.subn(r'FROZEN_MANIFEST_SHA256 = "[0-9a-f]*"',
                      f'FROZEN_MANIFEST_SHA256 = "{sha}"', src)
    if n1 != 1 or n2 != 1:
        print("could not find the pins in prespec_s2.py")
        return 1
    with open(path, "w") as fh:
        fh.write(src)
    print(f"code fingerprint {code}\nmanifest sha256  {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
