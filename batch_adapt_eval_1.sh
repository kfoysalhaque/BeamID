#!/usr/bin/env bash
set -euo pipefail

ROOT="Dataset/vmatrices"
K=55
REPEATS=1

# locations (0–8 in your case)
SOURCES=(1 2 3)
TARGETS=(1 2 3 4 5 6 7 8 9)

# checkpoint naming pattern
CKPT_DIR="."                 # change if checkpoints are in another folder
CKPT_PATTERN="checkpoint_s%s.pt"

OUTDIR="batch_results_k${K}"
mkdir -p "$OUTDIR"

for s in "${SOURCES[@]}"; do
  CKPT="$(printf "${CKPT_DIR}/${CKPT_PATTERN}" "$s")"
  if [[ ! -f "$CKPT" ]]; then
    echo "[warn] Missing checkpoint for source=$s: $CKPT (skipping this source)"
    continue
  fi

  OUT="${OUTDIR}/results_source_${s}.csv"

  # write header if file does not exist
  if [[ ! -f "$OUT" ]]; then
    echo "source,target,run,k,ckpt,same_env_acc,cross_no_adapt_acc,cross_after_adapt_acc" > "$OUT"
  fi

  for t in "${TARGETS[@]}"; do
    if [[ "$s" == "$t" ]]; then
      continue
    fi

    for run in $(seq 1 "$REPEATS"); do
      echo "[run] source=$s target=$t repeat=$run k=$K ckpt=$CKPT"

      # capture log without killing the whole script if one run fails
      if ! LOG=$(python adapt_eval.py "$ROOT" \
            --ckpt "$CKPT" \
            --source "$s" \
            --target "$t" \
            --k "$K" 2>&1); then
        echo "[err] adapt_eval failed for source=$s target=$t run=$run" >&2
        echo "$s,$t,$run,$K,$CKPT,NA,NA,NA" >> "$OUT"
        continue
      fi

      SAME=$(echo "$LOG"  | awk -F': ' '/Same-environment \(source val\) accuracy/{print $2}' | tail -n 1)
      NOAD=$(echo "$LOG"  | awk -F': ' '/Cross-environment \(no adapt\) accuracy/{print $2}' | tail -n 1)
      AFTER=$(echo "$LOG" | awk -F': ' '/Cross-environment \(few-shot FT\) acc/{print $2}' | tail -n 1)

      SAME=${SAME:-NA}
      NOAD=${NOAD:-NA}
      AFTER=${AFTER:-NA}

      echo "$s,$t,$run,$K,$CKPT,$SAME,$NOAD,$AFTER" >> "$OUT"
    done
  done
done

echo "[done] Results saved under $OUTDIR/"
