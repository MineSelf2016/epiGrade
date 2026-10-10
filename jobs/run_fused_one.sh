#!/bin/bash
cd "$(dirname "$0")/.."
read -r FILE NAME SPLIT FOLD ROOT SPLITS EXTRA <<< "$*"
KEY=$(echo "${ROOT}_${NAME}_${SPLIT}_${FOLD}_${EXTRA}" | tr ' /=.-' '_____')
mkdir "results_fused/.locks/$KEY" 2>/dev/null || exit 0
LODO=""; MODE="$SPLIT"
[ "$SPLIT" = "donor" ] && LODO="--lodo"
[ "$SPLIT" = "donor5" ] && MODE="donor"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
if .venv/bin/python -u code/experiments/run_fused.py --data "data/processed/${FILE}.h5ad" \
     --dataset-name "$NAME" --output-root "$ROOT" --split-mode "$MODE" $LODO \
     --splits "$SPLITS" --folds "$FOLD" $EXTRA > "logs/fused_${KEY}.out" 2>&1; then
  touch "results_fused/.locks/$KEY/done"
else
  echo "FAILED $KEY" >> logs/fused_failures.txt
  rm -rf "results_fused/.locks/$KEY"
fi
