"""Chọn lần chạy PhoBERT chính — bằng QUY TẮC, không bằng tay.

PhoBERT được chạy hai lần: lr 3e-5 (phân kỳ ở epoch 2, xem logs/train_phobert.log)
và lr 2e-5. Lần chạy chính là lần có F1 trên DEV (tách từ train) cao nhất — cùng
tiêu chí đã dùng để chọn epoch. Validation không tham gia vào lựa chọn này.
"""
from __future__ import annotations

import json
from pathlib import Path

PHOBERT_RUNS = ("phobert", "phobert_lr2e5")


def best_dev_f1(results: Path, run: str) -> float | None:
    p = results / f"training_curve_{run}.json"
    if not p.exists():
        return None
    curve = json.loads(p.read_text())["curve"]
    return max(e["val_f1"] for e in curve) if curve else None


def primary_phobert(results: Path) -> str:
    """Khoá tệp kết quả của lần chạy PhoBERT chính (có đủ dự đoán validation)."""
    scored = [(best_dev_f1(results, r), r) for r in PHOBERT_RUNS
              if (results / f"predictions_{r}_validation.json").exists()]
    scored = [(f, r) for f, r in scored if f is not None]
    return max(scored)[1] if scored else "phobert"
