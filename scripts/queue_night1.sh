#!/usr/bin/env bash
# Đêm 1: PhoBERT ổn định (lr 1e-5) rồi XLM-R seed 13, mỗi run chấm luôn full dev.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FILTER='warn|Loading weights|Writing model'
$PY scripts/finetune.py --model vinai/phobert-base-v2 --word-segmented --out models/phobert_stable \
  --epochs 3 --lr 1e-5 --seed 42 --split-seed 42 --max-length 256 --doc-stride 96 --max-answer-len 64 \
  2>&1 | grep -v -i -E "$FILTER" > results/logs/train_phobert_stable.log
$PY scripts/score_windows.py --checkpoint models/phobert_stable --name phobert_stable --split dev \
  2>&1 | grep -v -i -E "$FILTER" >> results/logs/score_windows.log
$PY scripts/finetune.py --model FacebookAI/xlm-roberta-base --out models/xlmr_seed13 \
  --epochs 3 --lr 3e-5 --seed 13 --split-seed 42 --max-length 384 --doc-stride 128 --max-answer-len 64 \
  2>&1 | grep -v -i -E "$FILTER" > results/logs/train_xlmr_seed13.log
$PY scripts/score_windows.py --checkpoint models/xlmr_seed13 --name xlmr_seed13 --split dev \
  2>&1 | grep -v -i -E "$FILTER" >> results/logs/score_windows.log
echo NIGHT1_DONE >> results/logs/score_windows.log
