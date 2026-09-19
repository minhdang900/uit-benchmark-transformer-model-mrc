"""Chọn lần chạy PhoBERT chính — bằng QUY TẮC, không bằng tay.

Hai lần chạy gốc (lr 3e-5 và lr 2e-5) đều sụp đổ ở lớp "không có đáp án" giữa
huấn luyện (``results/probe_resume.json``). Lần chạy gốc tốt nhất là lần có F1 cao
nhất trên DEV (tách từ train). Một lần chạy ỔN ĐỊNH (``phobert_stable``, lr 1e-5)
chỉ thay nó khi cả hai điều kiện đúng — quy tắc nhóm chọn trong spec:

* huấn luyện ổn định (``stability.stable`` trong training curve), và
* F1 trên toàn bộ dev CAO HƠN lần chạy hiện tại.

Nếu không, lần chạy cũ giữ vai trò chính và báo cáo phải nêu rõ nó đã sụp đổ.
Validation không tham gia vào lựa chọn này.
"""
from __future__ import annotations

import json
from pathlib import Path

PHOBERT_RUNS = ("phobert", "phobert_lr2e5")
STABLE_CANDIDATES = ("phobert_stable",)


def best_dev_f1(results: Path, run: str) -> float | None:
    """F1 dev tốt nhất trên đường cong huấn luyện (mẫu 500 câu)."""
    p = results / f"training_curve_{run}.json"
    if not p.exists():
        return None
    curve = json.loads(p.read_text())["curve"]
    return max(e["val_f1"] for e in curve) if curve else None


def full_dev_f1(results: Path, run: str) -> float | None:
    """F1 trên TOÀN BỘ dev ở τ = 0 (``scripts/score_windows.py``)."""
    p = results / f"windows_{run}_dev.json"
    return json.loads(p.read_text())["at_tau0"]["F1"] if p.exists() else None


def is_stable(results: Path, run: str) -> bool:
    p = results / f"training_curve_{run}.json"
    return bool(p.exists() and json.loads(p.read_text()).get("stability", {}).get("stable"))


def _dev_f1(results: Path, run: str) -> float | None:
    f = full_dev_f1(results, run)
    return f if f is not None else best_dev_f1(results, run)


def _has_predictions(results: Path, run: str) -> bool:
    return (results / f"predictions_{run}_validation.json").exists()


def current_phobert(results: Path) -> str:
    """Lần chạy gốc tốt nhất trên dev."""
    scored = [(_dev_f1(results, r), r) for r in PHOBERT_RUNS if _has_predictions(results, r)]
    scored = [(f, r) for f, r in scored if f is not None]
    return max(scored)[1] if scored else "phobert"


def primary_phobert(results: Path) -> str:
    """Lần chạy PhoBERT dùng trong mọi bảng chính của báo cáo."""
    current = current_phobert(results)
    bar = _dev_f1(results, current)
    for cand in STABLE_CANDIDATES:
        f = full_dev_f1(results, cand)
        if (_has_predictions(results, cand) and is_stable(results, cand)
                and f is not None and bar is not None and f > bar):
            return cand
    return current
