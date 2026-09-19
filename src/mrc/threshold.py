"""Ngưỡng "không có đáp án" τ, tách khỏi model để chọn τ mà không chạy lại GPU.

Với τ cố định bằng 0, tỉ lệ từ chối của hai model là sản phẩm phụ của cách chúng
được huấn luyện, và so sánh EM trên câu có đáp án trộn lẫn hai thứ: model đọc
giỏi đến đâu, và model sẵn lòng trả lời đến đâu. SQuAD 2.0 xử lý bằng cách chọn
τ riêng cho từng model trên dữ liệu giữ lại. Ở đây τ được chọn trên DEV (tách từ
train) rồi áp một lần lên validation.

Điểm từng cửa sổ do ``scripts/score_windows.py`` lưu; mỗi câu là một danh sách
``{"delta", "score", "answer"}`` với ``delta = null_score − best_score``.
"""
from __future__ import annotations

from typing import Mapping, Sequence

__all__ = ["decide", "predictions_at", "best_threshold"]


def decide(windows: Sequence[Mapping], tau: float) -> str:
    """Dự đoán của model ở ngưỡng ``tau`` — khớp ``TransformerQA.predict_detailed``."""
    best_score, answer = float("-inf"), ""
    for w in windows:
        delta = w["delta"]
        if delta is not None and delta + tau >= 0:
            continue  # cửa sổ này nói "không có đáp án"
        if w["score"] > best_score:
            best_score, answer = w["score"], w["answer"]
    return answer


def predictions_at(records: Mapping[str, Sequence[Mapping]], tau: float) -> dict[str, str]:
    return {qid: decide(ws, tau) for qid, ws in records.items()}


def _candidate_taus(records: Mapping[str, Sequence[Mapping]]) -> list[float]:
    # Quyết định chỉ đổi khi τ đi qua −delta của một cửa sổ. Thử 0 và điểm ngay
    # bên kia mỗi mốc là đủ để phủ mọi quyết định khác nhau.
    eps = 1e-6
    taus = {0.0}
    for ws in records.values():
        for w in ws:
            if w["delta"] is not None:
                taus.add(-w["delta"] + eps)
                taus.add(-w["delta"] - eps)
    return sorted(taus)


def best_threshold(
    records: Mapping[str, Sequence[Mapping]],
    references: Mapping[str, Sequence[str]],
) -> dict:
    """τ làm F1 lớn nhất trên ``records``; hoà thì chọn τ gần 0 nhất."""
    from mrc.metrics import evaluate

    best = None
    for tau in _candidate_taus(records):
        scores = evaluate(predictions_at(records, tau), references)
        key = (round(scores["F1"], 6), -abs(tau))
        if best is None or key > best[0]:
            best = (key, tau, scores)
    _, tau, scores = best
    return {"tau": round(tau, 6), "f1": scores["F1"], "em": scores["EM"]}
