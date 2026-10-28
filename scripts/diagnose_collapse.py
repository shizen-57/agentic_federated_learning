#!/usr/bin/env python3
"""Reproduce the FEMNIST collapse with diagnostics on (configs/diagnostic_v1.yaml).

    python scripts/diagnose_collapse.py --dry-run                  # what would run
    python scripts/diagnose_collapse.py --variant short --jobs 4   # 30-round runs
    python scripts/diagnose_collapse.py --variant full --jobs 4    # 200-round runs
    python scripts/report_diagnostics.py results/diagnostic_v1/short

Every run is expanded from the original s2 group by sweep.build_jobs, exactly
as the s2 runs were, with only fl.rounds, diag.enabled and the spec's shared
`set` (e.g. data.split) changed; --dry-run prints that difference for every
job, so nothing is approximated by hand.
The runs themselves go through scripts/sweep.py (resume, quarantine, and the
refusal to write into results/adv all apply). Each variant has its own
folder; a short run is never reused as, or padded into, a full one.

Writes <out>/<variant>/plan.yaml (the generated plan), MANIFEST.json (code
fingerprint, git commit and dirty state, plan hashes, package and device
versions, dataset identity, effective overrides) and sweep.log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import yaml  # noqa: E402

from afl.run import _load  # noqa: E402
from sweep import build_jobs, expected_config  # noqa: E402

# --set may only change HOW a screen runs, never WHAT it tests: anything else
# belongs in the spec (its `set`), where the report rebuilds it from
# (third external review, finding 2). The device is not part of the config check.
OPERATIONAL_OVERRIDES = ("fl.device",)


def non_operational(overrides: list[str]) -> list[str]:
    return [o for o in overrides if o.split("=", 1)[0].strip() not in OPERATIONAL_OVERRIDES]


def _sha(path: str, limit: int | None = None) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read() if limit is None else fh.read(limit))
    return h.hexdigest()


def _flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def plan_for(spec: dict, variant: str) -> tuple[dict, dict]:
    """(generated plan, source s2 group) for one variant."""
    src = _load(os.path.join(ROOT, spec["source_plan"]))
    g = src["groups"][spec["source_group"]]
    rounds = int(spec["variants"][variant])
    # spec "set": settings shared by every condition of this screen (e.g. the
    # persistent data split), applied on top of the s2 group's own
    group = {"config": g["config"], "set": {**(g.get("set") or {}), **(spec.get("set") or {}),
                                            "fl.rounds": rounds, "diag.enabled": True},
             "conditions": list(spec["conditions"]), "seeds": list(spec["seeds"])}
    return {"seeds": list(spec["seeds"]), "sets": {}, "groups": {spec["source_group"]: group}}, src


def compare(spec: dict, plan: dict, src: dict, out: str) -> list[tuple[dict, list[str]]]:
    """Each diagnostic job and the config keys in which it differs from the
    original s2 job with the same condition and seed."""
    mine = build_jobs(plan, None, None, out, [])
    orig = {(j["cond"], j["seed"]): j for j in
            build_jobs(src, [spec["source_group"]], list(spec["seeds"]), "ORIGINAL", [])}
    base_of = dict(spec.get("base_of", {}))    # revised condition -> the s2 one it modifies
    for c in spec.get("new_conditions", []):   # new methods: compared with the reference
        base_of.setdefault(c, spec["reference"])
    res = []
    for j in mine:
        a = _flat(expected_config(j["raw"]))
        b = _flat(expected_config(orig[(base_of.get(j["cond"], j["cond"]), j["seed"])]["raw"]))
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        res.append((j, diff))
    return res


def manifest(spec_path: str, spec: dict, variant: str, plan_path: str, overrides: list[str]) -> dict:
    import numpy as np
    import torch
    from afl.provenance import code_hash

    def git(*args):
        r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT)
        return r.stdout.strip() if r.returncode == 0 else None

    data = os.path.join(ROOT, "data")
    return {
        "variant": variant, "rounds": spec["variants"][variant], "label": "short diagnostic"
        if variant == "short" else "full-length diagnostic",
        "purpose": "failure reproduction on already-inspected seeds; not a test",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "code_hash": code_hash(), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "plan_sha256": {"diagnostic": _sha(spec_path), "generated": _sha(plan_path),
                        "source": _sha(os.path.join(ROOT, spec["source_plan"]))},
        "overrides_vs_source": ["fl.rounds", "diag.enabled", "out"]
                               + sorted(spec.get("set") or {}) + overrides,
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "cuda": torch.version.cuda,
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "dataset": {"femnist_y.npy": _sha(os.path.join(data, "femnist_y.npy")),
                    "femnist_writer.npy": _sha(os.path.join(data, "femnist_writer.npy")),
                    "femnist_x.npy_first_64MB": _sha(os.path.join(data, "femnist_x.npy"), 64 << 20),
                    "femnist_x.npy_bytes": (os.path.getsize(os.path.join(data, "femnist_x.npy"))
                                            if os.path.exists(os.path.join(data, "femnist_x.npy"))
                                            else None)},
    }


def sidecars_complete(raw: dict) -> str:
    """'ok' or what is missing from a run's diagnostic sidecars."""
    from afl.diagnostics import read
    base = os.path.join(raw["out"], raw["tag"])
    for k in ("round", "invite", "update"):
        path = f"{base}.diag_{k}.jsonl"
        if not os.path.exists(path):
            return f"no {k} sidecar"
    _, rounds = read(base + ".diag_round.jsonl")
    if [r["t"] for r in rounds] != list(range(1, raw["fl"]["rounds"] + 1)):
        return "round sidecar incomplete"
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(ROOT, "configs", "diagnostic_v1.yaml"))
    ap.add_argument("--variant", default="short")
    ap.add_argument("--out", default=None, help="default: the spec's out")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--set", action="append", default=[],
                    help="operational override passed to sweep.py, e.g. fl.device=auto (recorded); "
                         "experimental settings go in the spec's `set`")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.spec) as fh:
        spec = yaml.safe_load(fh)
    if args.variant not in spec["variants"]:
        raise SystemExit(f"unknown variant {args.variant}; have {list(spec['variants'])}")
    if non_operational(args.set):
        raise SystemExit(f"--set may only change {OPERATIONAL_OVERRIDES}; put "
                         f"{non_operational(args.set)} in the spec's `set` instead")
    import seeds as seedreg
    seedreg.check(spec["seeds"], spec.get("purpose", "diagnostic"), spec.get("frozen_manifest"))
    out = os.path.join(args.out or os.path.join(ROOT, spec["out"]), args.variant)
    plan, src = plan_for(spec, args.variant)
    rows = compare(spec, plan, src, out)

    print(f"diagnostic v{spec['version']} / {args.variant}: {len(rows)} runs of "
          f"{spec['variants'][args.variant]} rounds -> {out}")
    print("source: " + spec["source_plan"] + " group " + spec["source_group"]
          + " (same build_jobs expansion)")
    allowed = {"fl.rounds", "diag.enabled", "out", *spec.get("allowed_extra", []),
               *(spec.get("set") or {})}
    from afl.conditions import CONDITIONS
    for j, diff in rows:
        ok = allowed | ({"tag"} if j["cond"] in spec.get("base_of", {}) else set())
        if j["cond"] in spec.get("new_conditions", []):
            # a new method may differ from the reference only in what its own
            # condition sets (and its name) -- every s2 setting else is shared
            ok = ok | {"tag"} | set(CONDITIONS[j["cond"]])
        bad = [k for k in diff if k not in ok]
        flag = "  <-- UNEXPECTED" if bad else ""
        print(f"  {j['cond']:16s} seed {j['seed']}  {os.path.join(j['raw']['out'], j['raw']['tag'])}"
              f"  differs from s2 in: {', '.join(diff)}{flag}")
        if bad:
            raise SystemExit(f"unexpected differences from the s2 job: {bad}")
    if args.dry_run:
        return 0

    os.makedirs(out, exist_ok=True)
    plan_path = os.path.join(out, "plan.yaml")
    with open(plan_path, "w") as fh:
        yaml.safe_dump(plan, fh, sort_keys=True)
    with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
        json.dump(manifest(args.spec, spec, args.variant, plan_path, args.set), fh, indent=2)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "sweep.py"), "--plan", plan_path,
           "--out", out, "--jobs", str(args.jobs)] + [x for s in args.set for x in ("--set", s)]
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(os.path.join(out, "sweep.log"), "a") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                cwd=ROOT)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
    problems = [(j["raw"]["tag"], st) for j, _ in rows
                if (st := sidecars_complete(j["raw"])) != "ok"]
    print(f"wall {time.time() - t0:.0f}s; sidecars: "
          + ("all complete" if not problems else f"{len(problems)} incomplete: {problems}"))
    return rc or (1 if problems else 0)


if __name__ == "__main__":
    raise SystemExit(main())
