#!/usr/bin/env bash
# Đợt 2: chấm validation/stress cho các run mới, rồi PhoBERT seed 13 (cấu hình của run chính) và XLM-R 256/96.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FILTER='warn|Loading weights|Writing model|Token indices'
for ds in validation stress; do
  $PY scripts/run_eval.py --models phobert_stable xlmr_seed13 --dataset $ds 2>&1 | grep -v -i -E "$FILTER" >> results/logs/eval_night2.log
done
for run in phobert_stable xlmr_seed13; do
  $PY scripts/score_windows.py --checkpoint models/$run --name $run --split validation 2>&1 | grep -v -i -E "$FILTER" >> results/logs/score_windows.log
done
$PY scripts/finetune.py --model vinai/phobert-base-v2 --word-segmented --out models/phobert_seed13 \
  --epochs 3 --lr 3e-5 --seed 13 --split-seed 42 --max-length 256 --doc-stride 96 --max-answer-len 64 \
  2>&1 | grep -v -i -E "$FILTER" > results/logs/train_phobert_seed13.log
$PY scripts/score_windows.py --checkpoint models/phobert_seed13 --name phobert_seed13 --split dev 2>&1 | grep -v -i -E "$FILTER" >> results/logs/score_windows.log
$PY scripts/finetune.py --model FacebookAI/xlm-roberta-base --out models/xlmr_256 \
  --epochs 3 --lr 3e-5 --seed 42 --split-seed 42 --max-length 256 --doc-stride 96 --max-answer-len 64 \
  2>&1 | grep -v -i -E "$FILTER" > results/logs/train_xlmr_256.log
$PY scripts/score_windows.py --checkpoint models/xlmr_256 --name xlmr_256 --split dev 2>&1 | grep -v -i -E "$FILTER" >> results/logs/score_windows.log
for ds in validation stress; do
  $PY scripts/run_eval.py --models phobert_seed13 xlmr_256 --dataset $ds 2>&1 | grep -v -i -E "$FILTER" >> results/logs/eval_night2.log
done
for run in phobert_seed13 xlmr_256; do
  $PY scripts/score_windows.py --checkpoint models/$run --name $run --split validation 2>&1 | grep -v -i -E "$FILTER" >> results/logs/score_windows.log
done
echo NIGHT2_DONE >> results/logs/score_windows.log
