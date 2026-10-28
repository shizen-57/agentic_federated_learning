#!/usr/bin/env bash
# Run the project on the department GPU server (or any Linux GPU machine).
# Run every step from the repository root, inside tmux so a lost connection
# does not stop the runs.
#
#   bash scripts/dept_gpu.sh setup      # once: own conda env + PyTorch + data
#   bash scripts/dept_gpu.sh check      # GPU visible? cores? one quick timing
#   bash scripts/dept_gpu.sh repro113   # FEMNIST seed 113, all 23 methods (~10 min)
#   bash scripts/dept_gpu.sh lockseeds  # the other lock-up seeds 103 110 112 116 (~30 min)
#   bash scripts/dept_gpu.sh diag short # collapse diagnostics, 8 runs x 30 rounds
#   bash scripts/dept_gpu.sh diag full  # the same 8 runs at the full 200 rounds
#   bash scripts/dept_gpu.sh pretrain femnist ppo_full   # RL2 pre-training -> checkpoints/
#   bash scripts/dept_gpu.sh screen configs/rl2_screen_b.yaml   # a development screen
#   bash scripts/dept_gpu.sh tune       # RL2 M7: tuning (docs/RL2_M7_TUNING.md)
#   bash scripts/dept_gpu.sh screen-c-pretrain   # RL2 M8: the checkpoints screen C needs
#   bash scripts/dept_gpu.sh screen-c   # RL2 M8: screen C (docs/RL2_SCREEN_C.md)
#   bash scripts/dept_gpu.sh confirm    # RL2 M10: checks only; "confirm go" runs it (user's go-ahead)
#   bash scripts/dept_gpu.sh screen2 configs/rl2_arl_screen.yaml   # a controlled screen (ARL, sensitivity)
#   bash scripts/dept_gpu.sh devicecheck  # short real-data runs to compare with the CPU
#   bash scripts/dept_gpu.sh packsmall rl2_tune   # small results file to send (no big logs)
#   bash scripts/dept_gpu.sh pack       # pack the results to copy back home
#
# Server rules this script follows: everything goes into our own conda env
# ("afl"), never the base env; NVIDIA drivers are never touched; results are
# written under runs/ and must be copied off the machine after each session.
set -euo pipefail

ENV=afl
OUT=runs
JOBS=${JOBS:-}

activate() {
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV"
    in_own_env
    # less GPU memory lost to fragmentation when several runs share the card
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

# Server rule: never install into (or run from) the base environment.
in_own_env() {
    local base; base="$(conda info --base)"
    if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ] || [ -z "${CONDA_PREFIX:-}" ] \
       || [ "$CONDA_PREFIX" = "$base" ] \
       || [ "$(python -c 'import sys; print(sys.prefix)')" != "$CONDA_PREFIX" ]; then
        echo "STOP: not inside the '$ENV' conda env (python is $(command -v python)); nothing installed."
        exit 1
    fi
}

jobs_default() {
    if [ -z "$JOBS" ]; then
        local n; n=$(nproc)
        JOBS=$(( n < 4 ? n : 4 ))
    fi
}

case "${1:-}" in
setup)
    command -v conda >/dev/null || { echo "conda not found: load Anaconda first (e.g. 'module load anaconda')"; exit 1; }
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda env list | grep -q "^$ENV " || conda create -y -n "$ENV" python=3.11
    conda activate "$ENV"
    in_own_env
    # Pick a PyTorch build the installed driver supports; never change the driver.
    cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
    if [ -z "$cuda" ]; then echo "nvidia-smi shows no CUDA version"; exit 1; fi
    if   [ "$cuda" -ge 121 ]; then idx=cu121
    elif [ "$cuda" -ge 118 ]; then idx=cu118
    else echo "Driver supports CUDA ${cuda:0:2}.${cuda:2}: too old for current PyTorch. Ask the admin; do not update the driver yourself."; exit 1
    fi
    # pip of the env's own python, so packages can only land in the env
    python -m pip install --index-url "https://download.pytorch.org/whl/$idx" torch
    python -m pip install numpy pyyaml matplotlib pytest pyarrow pillow pandas
    python scripts/prepare_femnist.py
    PYTHONPATH=src python -c "from afl.data import ensure; ensure('cifar10')"
    echo "setup done"
    ;;
check)
    activate
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
    echo "CPU cores: $(nproc)"
    python -c "import torch; print('torch', torch.__version__, '| cuda ok:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
    python - <<'EOF'
import sys, time
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from sweep import build_jobs
from afl.run import _load
from afl.config import Cfg
from afl.engine import Experiment
for g in ("s2_femnist", "s2_cifar"):
    job = build_jobs(_load("configs/sweep_advanced.yaml"), [g], [0], "/tmp/afl_check", ["fl.device=auto"])[0]
    raw = job["raw"]; rounds = raw["fl"]["rounds"]; raw["fl"]["rounds"] = 3
    t0 = time.time(); exp = Experiment(Cfg.from_dict(raw)); setup = time.time() - t0
    t0 = time.time(); exp.run(verbose=False); per = (time.time() - t0) / 3
    print(f"{g}: device {exp.dev}, {per:.2f} s/round -> one full run ~{(setup + per * rounds) / 60:.1f} min alone")
EOF
    ;;
repro113)
    activate; jobs_default; mkdir -p "$OUT"
    python scripts/sweep.py --groups s2_femnist --seeds 113 --out "$OUT/repro113" \
        --jobs "$JOBS" --set fl.device=auto 2>&1 | tee -a "$OUT/repro113.log"
    ;;
lockseeds)
    activate; jobs_default; mkdir -p "$OUT"
    python scripts/sweep.py --groups s2_femnist --seeds 103 110 112 116 --out "$OUT/lockseeds" \
        --jobs "$JOBS" --set fl.device=auto 2>&1 | tee -a "$OUT/lockseeds.log"
    ;;
diag)
    activate; jobs_default; mkdir -p "$OUT"
    python scripts/diagnose_collapse.py --variant "${2:-short}" --jobs "$JOBS" \
        --set fl.device=auto --out "$OUT/diagnostic_v1" 2>&1 | tee -a "$OUT/diag.log"
    python scripts/report_diagnostics.py "$OUT/diagnostic_v1/${2:-short}" \
        > "$OUT/diagnostic_v1/${2:-short}/report.md"
    echo "report: $OUT/diagnostic_v1/${2:-short}/report.md"
    ;;
screen)
    # a pre-registered development screen: bash scripts/dept_gpu.sh screen configs/<spec>.yaml
    activate; jobs_default; mkdir -p "$OUT"
    spec=${2:?give the screen spec, e.g. configs/rl2_screen_b.yaml}
    name=$(basename "$spec" .yaml)
    # pre-flight: every condition runs 2 rounds on this machine first (~1 min)
    python scripts/smoke_screen.py --spec "$spec" --set fl.device=auto \
        || { echo "STOP: a condition crashed in the 2-round check; send the output above"; exit 1; }
    python scripts/diagnose_collapse.py --spec "$spec" --variant full --jobs "$JOBS" \
        --set fl.device=auto --out "$OUT/$name" 2>&1 | tee -a "$OUT/$name.log"
    # the pre-registered rule, applied only if every expected run is complete and valid
    group=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["source_group"])' "$spec")
    python scripts/report_dev_screen.py "$OUT/$name/full/$group" --spec "$spec" \
        > "$OUT/$name/full/report.md" \
        || echo "report says INCOMPLETE -- no decision (see $OUT/$name/full/report.md)"
    echo "report: $OUT/$name/full/report.md"
    ;;
pretrain)
    activate; mkdir -p checkpoints
    ds=${2:-femnist}; cond=${3:-ppo_full}
    python scripts/pretrain.py --group "s2_$ds" --condition "$cond" --iterations "${ITER:-20}" \
        --workers "${WORKERS:-8}" --set fl.device=auto --out "checkpoints/${cond}_${ds}.pt" \
        2>&1 | tee -a "$OUT/pretrain_${cond}_${ds}.log"
    ;;
tune)
    activate; jobs_default; mkdir -p "$OUT"
    python scripts/tune_rl2.py --spec configs/rl2_tune.yaml --out "$OUT/rl2_tune" --jobs "$JOBS" \
        --set fl.device=auto 2>&1 | tee -a "$OUT/rl2_tune.log"
    ;;
screen-c-pretrain)
    activate; mkdir -p "$OUT"
    # resumable: run it again after an interruption. MAXPROCS=2 if the GPU runs out of memory.
    python scripts/screen_c_rl2.py pretrain --set fl.device=auto ${MAXPROCS:+--max-procs "$MAXPROCS"} \
        2>&1 | tee -a "$OUT/rl2_screen_c_pretrain.log"
    ;;
screen-c)
    activate; JOBS=${JOBS:-3}; mkdir -p "$OUT"     # 3: four CIFAR-10 runs do not fit the 8 GB GPU
    python scripts/screen_c_rl2.py run --out "$OUT/rl2_screen_c" --jobs "$JOBS" --set fl.device=auto \
        2>&1 | tee -a "$OUT/rl2_screen_c.log"
    ;;
confirm)
    # without "go": every check, nothing run. "confirm go" = the user's go-ahead.
    activate; JOBS=${JOBS:-3}; mkdir -p "$OUT"     # 3: four CIFAR-10 runs do not fit the 8 GB GPU
    python scripts/confirm_rl2.py --out "$OUT/rl2_confirm" --jobs "$JOBS" --set fl.device=auto \
        ${MAXPROCS:+--max-procs "$MAXPROCS"} $( [ "${2:-}" = go ] && echo --go ) 2>&1 | tee -a "$OUT/rl2_confirm.log"
    ;;
screen2)
    # a controlled development screen: bash scripts/dept_gpu.sh screen2 configs/<spec>.yaml
    activate; JOBS=${JOBS:-3}; mkdir -p "$OUT"     # 3: CIFAR-10 runs need ~1.8 GB each
    spec=${2:?give the spec, e.g. configs/rl2_arl_screen.yaml}
    name=$(basename "$spec" .yaml)
    python scripts/screen_rl2.py run --spec "$spec" --out "$OUT/$name" --jobs "$JOBS" --set fl.device=auto \
        2>&1 | tee -a "$OUT/$name.log"
    ;;
devicecheck)
    activate; mkdir -p "$OUT/devicecheck"
    python scripts/check_devices.py --out "$OUT/devicecheck/gpu.json" --set fl.device=auto 2>&1 | tee -a "$OUT/devicecheck.log"
    ;;
packsmall)
    # what Claude needs to check a stage: tables, configs, manifests, reports and
    # the (small) checkpoints; no diagnostic sidecars, no quarantined runs
    name=${2:?give the runs folder, e.g. rl2_tune}
    tar --exclude='*.diag_*' --exclude='*/quarantine/*' -czf "${name}_small.tar.gz" "$OUT/$name" \
        $( [ -d checkpoints ] && echo checkpoints )
    ls -la "${name}_small.tar.gz"
    echo "Copy it home (on YOUR computer): scp -P <port> <user>@<server>:$(pwd)/${name}_small.tar.gz ."
    ;;
pack)
    stamp=$(date +%Y%m%d_%H%M)
    tar -czf "results_$stamp.tar.gz" "$OUT" $( [ -d checkpoints ] && echo checkpoints )
    ls -la "results_$stamp.tar.gz"
    echo "Now copy it home (run this on YOUR computer):"
    echo "  scp -P <port> <user>@<server>:$(pwd)/results_$stamp.tar.gz ."
    ;;
*)
    sed -n '2,27p' "$0"; exit 1
    ;;
esac
