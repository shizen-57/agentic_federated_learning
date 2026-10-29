"""Shared pieces of the RL2 pipeline after screen B -- M7 tuning
(tune_rl2.py), M8 screen C (screen_c_rl2.py), M9 freeze (freeze_rl2.py) and
M10 confirmation (confirm_rl2.py, prespec_rl2.py): plan groups built from an
s2 group, run manifests, the evidence check, a pre-flight smoke run, the
endpoint score and the paired tests. One implementation, so every stage is
run and judged the same way (docs/RL2_M7_TUNING.md, docs/RL2_SCREEN_C.md,
docs/RL2_CONFIRMATION.md).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from afl.run import _load  # noqa: E402
from diagnose_collapse import _sha, non_operational  # noqa: E402

GROUP_VALUE = "__group__"      # an override value meaning "the group's own value"
SMOKE_SEED = 900               # GPU-smoke seed (configs/seed_registry.yaml)
LABELS = {"diagnostic": "spec", "spec": "spec", "source": "source plan", "generated": "generated plan"}


# ----------------------------------------------------------------- plans

def source_group(plan_path: str, group: str) -> dict:
    return _load(os.path.join(ROOT, plan_path))["groups"][group]


def group_entry(src: dict, shared: dict, rounds: int, conditions, seeds, override: dict | None = None,
                diag: bool = False) -> dict:
    """A plan group: the s2 group's config and settings, the stage's shared
    settings (e.g. the persistent split), and optionally an `override` that is
    applied after the condition (sweep.build_jobs)."""
    entry = {"config": src["config"],
             "set": {**(src.get("set") or {}), **(shared or {}), "fl.rounds": int(rounds),
                     "diag.enabled": bool(diag)},
             "conditions": list(conditions), "seeds": [int(s) for s in seeds]}
    if override:
        entry["override"] = dict(override)
    return entry


def group_defaults(entry: dict) -> dict:
    """The full config of a group before any condition is applied."""
    from afl.conditions import apply
    from afl.config import Cfg
    raw = _load(os.path.join(ROOT, entry["config"]))
    apply(raw, entry.get("set") or {})
    return Cfg.from_dict(raw).to_dict()


def resolve(entry: dict, overrides: dict) -> dict:
    """Replace GROUP_VALUE ("__group__") by the group's own value of that key,
    so "switch the cost constraint off" restores the fixed cost weight the
    group itself uses."""
    if not any(v == GROUP_VALUE for v in overrides.values()):
        return dict(overrides)
    base = group_defaults(entry)
    out = {}
    for k, v in overrides.items():
        if v == GROUP_VALUE:
            node = base
            for part in k.split("."):
                node = node[part]
            v = node
        out[k] = v
    return out


def check_pool(plan: dict, purpose: str) -> None:
    """Development runs may never measure on the confirmation pool, and the
    confirmation must (data.eval_pool; docs/RL2_CONFIRMATION.md)."""
    from sweep import build_jobs
    want = "confirm" if purpose == "confirmation" else "dev"
    for j in build_jobs(plan, None, None, "/x", []):
        d = j["raw"].get("data", {})
        got = d.get("eval_pool", "dev") if d.get("split") == "persistent" else None
        if purpose == "confirmation" and got != "confirm":
            raise SystemExit(f"{j['group']}/{j['cond']}: the confirmation must use data.eval_pool=confirm")
        if purpose != "confirmation" and got == "confirm":
            raise SystemExit(f"{j['group']}/{j['cond']}: a {purpose} run may not use the confirmation pool")
        if got is not None and got != want:
            raise SystemExit(f"{j['group']}/{j['cond']}: data.eval_pool {got}, expected {want}")


def write_plan(plan: dict, out: str) -> str:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "plan.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(plan, fh, sort_keys=True)
    return path


# -------------------------------------------------------------- manifests

def git(*args) -> str | None:
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip() if r.returncode == 0 else None


def write_manifest(out: str, inputs: dict[str, str], overrides: list[str], **extra) -> dict:
    """MANIFEST.json for one stage: code fingerprint, git commit and clean
    state, the SHA-256 of every input (spec, source plan, generated plan,
    tuned config, checkpoints...), runner overrides, library versions."""
    import numpy as np
    import torch
    from afl.provenance import code_hash
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "code_hash": code_hash(),
           "git_commit": git("rev-parse", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "inputs": {k: _sha(p) for k, p in inputs.items()},
           "generated": _sha(os.path.join(out, "plan.yaml")),
           "overrides_vs_source": list(overrides),
           "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", **extra}
    with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    return man


# ---------------------------------------------------------------- running

def run_sweep(plan_path: str, out: str, jobs: int, sets: list[str]) -> int:
    """Run (or resume) a plan through scripts/sweep.py, log in out/sweep.log."""
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "sweep.py"), "--plan", plan_path,
           "--out", out, "--jobs", str(jobs)] + [x for s in sets for x in ("--set", s)]
    print("running:", " ".join(cmd), flush=True)
    with open(os.path.join(out, "sweep.log"), "a") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return proc.wait()


def smoke(plan: dict, sets: list[str], rounds: int = 2) -> list[str]:
    """Run every (group, condition) of a plan for a few rounds on THIS
    machine (seed 900, temporary folder, nothing read). Returns the failures."""
    from afl.config import Cfg
    from afl.engine import Experiment
    from sweep import build_jobs
    if non_operational(sets):
        raise SystemExit(f"--set may only be operational, not {non_operational(sets)}")
    tmp = tempfile.mkdtemp(prefix="afl_smoke_")
    failed, seen = [], set()
    try:
        for j in build_jobs(plan, None, [SMOKE_SEED], tmp,
                            [f"fl.rounds={rounds}", "diag.enabled=false", *sets]):
            key = f"{j['group']}/{j['cond']}"
            if key in seen:
                continue
            seen.add(key)
            t0 = time.time()
            try:
                Experiment(Cfg.from_dict(j["raw"])).run(verbose=False)
                print(f"ok      {key:40s} {time.time() - t0:5.1f}s", flush=True)
            except Exception as exc:
                failed.append(key)
                print(f"FAILED  {key:40s} {type(exc).__name__}: {exc}", flush=True)
                if len(failed) == 1:
                    traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"smoke check: {len(seen) - len(failed)}/{len(seen)} ran"
          + (f"; FAILED {failed} -- do not start the runs" if failed else ""), flush=True)
    return failed


# ---------------------------------------------------------- evidence check

def check_evidence(out: str, plan: dict, inputs: dict[str, str],
                   manifest_key: str = "inputs") -> tuple[list[str], list[dict], list[str]]:
    """(problems, expected jobs, CSV paths of the valid runs). Every run the
    plan asks for must exist with exactly the expected config (the output
    folder aside), all rounds, finite values, no source change mid-run; all
    from one code version -- the one in out/MANIFEST.json, made from a clean
    checkout, and the one running this check; the inputs and plan.yaml must
    match their recorded hashes; runner overrides only operational."""
    from afl.provenance import code_hash
    from sweep import build_jobs, run_status
    problems: list[str] = []
    try:
        with open(os.path.join(out, "MANIFEST.json")) as fh:
            man = json.load(fh)
    except (OSError, ValueError):
        return [f"no readable MANIFEST.json in {out}"], [], []
    if man.get("git_dirty") is not False:
        problems.append("the runs were made from a checkout with uncommitted changes (or unknown)")
    if man.get("code_hash") != code_hash():
        return problems + [f"this checkout's simulator code ({code_hash()}) is not the code that "
                           f"made the runs ({man.get('code_hash')}, commit {man.get('git_commit')}); "
                           "run the report from that commit"], [], []
    shas = man.get(manifest_key) or {}
    plan_path = os.path.join(out, "plan.yaml")
    recorded_plan = shas.get("generated", man.get("generated"))
    for name, path in [*inputs.items(), ("generated", plan_path)]:
        want = recorded_plan if name == "generated" else shas.get(name)
        if _sha(path) is None or _sha(path) != want:
            problems.append(f"the {LABELS.get(name, name)} ({path}) does not match the hash "
                            "recorded in MANIFEST.json")
    try:
        with open(plan_path) as fh:
            if yaml.safe_load(fh) != plan:
                problems.append("plan.yaml is not the plan this spec generates")
    except (OSError, yaml.YAMLError):
        problems.append("no readable plan.yaml")
    bad = non_operational([o for o in man.get("overrides_vs_source", []) if "=" in o])
    if bad:
        problems.append(f"the runs used overrides that change the experiment: {bad} "
                        "(only the spec may set those)")
    jobs = build_jobs(plan, None, None, out, [])
    many = len(plan["groups"]) > 1
    hashes, valid = set(), []
    for j in jobs:
        # "out" is where the files are, which is checked by finding them there
        status = run_status(j["raw"], ignore=("out",))
        if status != "ok":
            problems.append(f"{j['group'] + '/' if many else ''}{j['cond']} seed {j['seed']}: {status}")
            continue
        base = os.path.join(j["raw"]["out"], j["raw"]["tag"])
        valid.append(base + ".csv")
        with open(base + ".config.json") as fh:
            hashes.add(json.load(fh).get("code_hash"))
    if len(hashes) > 1:
        problems.append(f"runs made by different code versions: {sorted(map(str, hashes))}")
    elif hashes and hashes != {man.get("code_hash")}:
        problems.append(f"runs' code {hashes.pop()} != MANIFEST code {man.get('code_hash')}")
    return problems, jobs, valid


def measures_by_group(jobs: list[dict]) -> dict[str, dict[str, dict[int, dict]]]:
    """{group: {condition: {seed: measures}}} for the given (valid) jobs."""
    import report_dev_screen as rep
    out: dict = {}
    for j in jobs:
        path = os.path.join(j["raw"]["out"], j["raw"]["tag"] + ".csv")
        out.setdefault(j["group"], {}).setdefault(j["cond"], {})[j["seed"]] = rep.measures(path)
    return out


# ----------------------------------------------------------------- scores

def score(m: dict) -> float:
    """The number M7 tuning ranked by (and the matched baseline tuning, for
    parity): the mean of equal-spend accuracy (eq5) and worst-10% client
    accuracy over the last 10 rounds (w10, whatever was spent); a missing
    value counts as 0."""
    eq = m["eq5"] if m["eq5"] == m["eq5"] else 0.0
    w = m["w10"] if m["w10"] == m["w10"] else 0.0
    return (eq + w) / 2.0


def score_eq(m: dict) -> float:
    """From screen C on (gap review A2): both endpoints at equal spend -- eq5
    and w10b, worst-10% client accuracy over the last 10 rounds within the
    same budget; a missing value counts as 0."""
    eq = m["eq5"] if m["eq5"] == m["eq5"] else 0.0
    w = m["w10b"] if m["w10b"] == m["w10b"] else 0.0
    return (eq + w) / 2.0


# ------------------------------------------------------------------ stats

def signflip_p(d, alternative: str = "greater", draws: int = 200_000, seed: int = 20261028) -> float:
    """Paired sign-flip p-value of mean(d) (exact over all 2^n sign patterns
    for n <= 20, else Monte Carlo with a fixed seed, p = (k+1)/(draws+1)).
    alternative: "greater" (mean > 0) or "two-sided"."""
    import numpy as np
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan")
    if n <= 20:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        stats = signs @ d
        exact = True
    else:
        rng = np.random.default_rng(seed)
        stats = rng.choice((-1.0, 1.0), size=(draws, n)) @ d
        exact = False
    obs = d.sum()
    if alternative == "greater":
        k = int((stats >= obs - 1e-12).sum())
    elif alternative == "two-sided":
        k = int((np.abs(stats) >= abs(obs) - 1e-12).sum())
    else:
        raise ValueError(alternative)
    return k / len(stats) if exact else (k + 1) / (len(stats) + 1)


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as given)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def seeds_needed(sd: float, delta: float, tests: int, power: float = 0.8, alpha: float = 0.05) -> int:
    """Paired seeds for a one-sided test of a mean difference delta with
    paired SD sd, Bonferroni-safe for `tests` tests (Holm is never less
    powerful): n = ((z_{1-alpha/tests} + z_power) * sd / delta)^2."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    import math
    return max(2, math.ceil(((z(1 - alpha / tests) + z(power)) * sd / delta) ** 2))

