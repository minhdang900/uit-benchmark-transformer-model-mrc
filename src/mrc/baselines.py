"""Baseline "luôn từ chối trả lời".

Không phải một model — một THƯỚC ĐO. Theo quy ước chấm SQuAD 2.0, câu impossible
được tính đúng khi dự đoán rỗng. Vì vậy một hệ thống không làm gì cả vẫn đạt điểm
bằng đúng tỉ lệ câu impossible của tập chấm (32,39% ở train, 30,44% ở validation
của ViQuAD 2.0). Mọi con số của model thật phải được đọc SO VỚI mức sàn này, và
nhóm câu 100% impossible (nhóm E3 của bộ stress-test) cho nó điểm tuyệt đối.
"""

from __future__ import annotations

from mrc.predictor import TimedPredictorMixin

__all__ = ["AlwaysAbstain"]


class AlwaysAbstain(TimedPredictorMixin):
    name = "Always-abstain"

    def predict(self, context: str, question: str) -> str:
        return ""
