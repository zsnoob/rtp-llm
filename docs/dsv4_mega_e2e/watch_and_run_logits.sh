#!/usr/bin/env bash
# Wait for a GPU with >=200GB free, then run the three-step Flash logits
# compare (baseline -> mega -> compare) on it. Poll every 5 minutes.
set -u
CKPT=${E2E_CKPT:?set E2E_CKPT to the checkpoint dir}
SCRIPT="$(dirname "$(readlink -f "$0")")/run_e2e_logits.py"
LOG=${E2E_WATCH_LOG:-/tmp/dsv4_logits_run.log}

echo "[watch] $(date) start" > "$LOG"
while true; do
    CANDIDATE=$(python3 - <<'PY'
import torch

if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        if capability != (10, 3):
            continue
        free, total = torch.cuda.mem_get_info(index)
        used_mib = (total - free) // (1024 * 1024)
        if used_mib < 40000:
            print(index, used_mib)
            break
PY
    )
    read -r GPU USED <<< "${CANDIDATE:-}"
    if [ -n "${GPU:-}" ]; then
        # Double-check after a short settle to avoid racing another job's startup.
        sleep 60
        CANDIDATE=$(CUDA_VISIBLE_DEVICES="$GPU" python3 - <<'PY'
import torch

if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 3):
    free, total = torch.cuda.mem_get_info(0)
    used_mib = (total - free) // (1024 * 1024)
    if used_mib < 40000:
        print(used_mib)
PY
        )
        if [ -n "${CANDIDATE:-}" ]; then
            USED=$CANDIDATE
            echo "[watch] $(date) claiming GPU $GPU (used=${USED}MiB)" >> "$LOG"
            break
        fi
    fi
    echo "[watch] $(date) no free GPU yet" >> "$LOG"
    sleep 300
done

cd "$(dirname "$SCRIPT")"
for MODE in baseline mega compare; do
    echo "[watch] $(date) mode=$MODE" >> "$LOG"
    E2E_CKPT=$CKPT E2E_GPU=$GPU python3 "$SCRIPT" "$MODE" >> "$LOG" 2>&1
    RC=$?
    echo "[watch] mode=$MODE rc=$RC" >> "$LOG"
    if [ $RC -ne 0 ] && [ "$MODE" != "compare" ]; then
        echo "[watch] aborting after $MODE failure" >> "$LOG"
        exit $RC
    fi
done
echo "[watch] $(date) all done" >> "$LOG"
