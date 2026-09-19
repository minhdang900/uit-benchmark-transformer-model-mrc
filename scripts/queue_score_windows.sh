#!/usr/bin/env bash
# Chấm điểm từng cửa sổ cho mọi checkpoint cần thiết (tuần tự — MPS chỉ một job).
set -u
cd "$(dirname "$0")/.."
while read -r ckpt name split; do
  [ -z "$ckpt" ] && continue
  .venv/bin/python scripts/score_windows.py --checkpoint "$ckpt" --name "$name" --split "$split" \
    2>&1 | grep -v -i "warn\|Loading weights"
done <<JOBS
${JOBS:-models/phobert phobert validation
models/phobert/epoch1 phobert_epoch1 dev
models/phobert phobert dev
models/xlmr xlmr dev
models/xlmr xlmr validation
models/phobert_lr2e5 phobert_lr2e5 dev
models/phobert_lr2e5 phobert_lr2e5 validation}
JOBS
echo DONE
