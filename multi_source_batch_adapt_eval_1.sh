#!/usr/bin/env bash
set -euo pipefail

ROOT="Dataset/vmatrices"
K=25
REPEATS=1

# All locations in your dataset
ALL_LOCS=(1 2 3 4 5 6 7 8 9)

CKPT_DIR="."
OUTDIR="batch_results_k${K}_source_sweep_unfreeze_last"
mkdir -p "$OUTDIR"

# helper: join array with "_"
join_by_underscore () {
  local IFS="_"
  echo "$*"
}

# Sweep number of source domains
# assuming checkpoints start from 1_2_3 and grow
for n_src in $(seq 2 8); do
  SOURCES=("${ALL_LOCS[@]:0:${n_src}}")
  TARGETS=("${ALL_LOCS[@]:${n_src}}")

  SRC_TAG=$(join_by_underscore "${SOURCES[@]}")
  CKPT="${CKPT_DIR}/checkpoint_s${SRC_TAG}.pt"

  if [[ ! -f "$CKPT" ]]; then
    echo "[warn] Missing checkpoint: $CKPT (skipping)"
    continue
  fi

  OUT="${OUTDIR}/results_sources_${SRC_TAG}.csv"

  if [[ ! -f "$OUT" ]]; then
    echo "n_source,source_set,target,run,k,ckpt,same_env_acc,cross_no_adapt_acc,cross_after_adapt_acc" > "$OUT"
  fi

  echo "===================================================="
  echo "[source sweep] sources=(${SOURCES[*]})"
  echo "              targets=(${TARGETS[*]})"
  echo "              ckpt=$CKPT"
  echo "===================================================="

  for t in "${TARGETS[@]}"; do
    for run in $(seq 1 "$REPEATS"); do
      echo "[run] n_source=$n_src target=$t run=$run"

      if ! LOG=$(python adapt_eval.py "$ROOT" \
          --ckpt "$CKPT" \
          --source "${SOURCES[@]}" \
          --target "$t" \
          --unfreeze_last \
          --k "$K" 2>&1); then
        echo "[err] failed for sources=(${SOURCES[*]}) target=$t"
        echo "$n_src,\"$SRC_TAG\",$t,$run,$K,$CKPT,NA,NA,NA" >> "$OUT"
        continue
      fi

      SAME=$(echo "$LOG"  | awk -F': ' '/Same-environment \(source val\) accuracy/{print $2}')
      NOAD=$(echo "$LOG"  | awk -F': ' '/Cross-environment \(no adapt\) accuracy/{print $2}')
      AFTER=$(echo "$LOG" | awk -F': ' '/Cross-environment \(few-shot FT\) acc/{print $2}')

      echo "$n_src,\"$SRC_TAG\",$t,$run,$K,$CKPT,$SAME,$NOAD,$AFTER" >> "$OUT"
    done
  done
done

echo "[done] Results saved under $OUTDIR/"
