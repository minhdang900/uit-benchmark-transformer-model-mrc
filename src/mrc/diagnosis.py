"""Phân tích chẩn đoán trên dự đoán TỪNG CÂU — hàm thuần, test được không cần GPU.

Ba câu hỏi mà điểm tổng không trả lời được:

1. **Sai kiểu gì?** :func:`classify_error` xếp mỗi lỗi vào một trong sáu loại
   loại trừ nhau, định nghĩa bằng quan hệ tập hợp giữa token dự đoán và token vàng
   — không cần người gán, nên tái lập được trên toàn bộ tập chấm.
2. **Biết từ chối đúng lúc không?** :func:`abstention_stats` coi "trả rỗng" là một
   bộ phân loại nhị phân và đo precision/recall của nó.
3. **Hai model khác nhau thật hay do nhiễu?** :func:`paired_comparison` dùng kiểm
   định McNemar (chính xác) trên các câu mà hai model bất đồng, và bootstrap cho
   khoảng tin cậy của chênh lệch EM. Hai model chấm trên CÙNG câu hỏi nên phép so
   sánh cặp mạnh hơn nhiều so với so hai con số tổng.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence

from mrc.metrics import exact_match, token_f1
from mrc.normalize import tokenize

__all__ = [
    "ERROR_TYPES",
    "abstention_stats",
    "classify_error",
    "error_taxonomy",
    "mcnemar_exact",
    "paired_comparison",
    "slice_scores",
]

#: Thứ tự cố định để bảng và hình trong báo cáo luôn khớp nhau.
ERROR_TYPES = (
    "false_abstain",      # câu có đáp án, model trả rỗng
    "false_answer",       # câu impossible, model vẫn trả lời
    "boundary_superset",  # dự đoán CHỨA đáp án vàng, thừa chữ
    "boundary_subset",    # dự đoán NẰM TRONG đáp án vàng, thiếu chữ
    "boundary_overlap",   # giao nhau một phần, lệch cả hai phía
    "wrong_span",         # không có token chung — chọn sai chỗ
)


def classify_error(prediction: str, golds: Sequence[str]) -> str | None:
    """Loại lỗi của một dự đoán; ``None`` nếu dự đoán đúng (EM = 1).

    ``golds`` rỗng nghĩa là câu impossible. Với nhiều đáp án vàng, dùng đáp án có F1
    cao nhất với dự đoán — cùng quy ước "max over ground truths" của SQuAD.
    """
    pred_empty = not tokenize(prediction)
    if not golds:
        return None if pred_empty else "false_answer"
    if any(exact_match(prediction, g) for g in golds):
        return None
    if pred_empty:
        return "false_abstain"

    gold = max(golds, key=lambda g: token_f1(prediction, g))
    p, g = tokenize(prediction), tokenize(gold)
    if not (Counter(p) & Counter(g)):
        return "wrong_span"
    p_str, g_str = " ".join(p), " ".join(g)
    if g_str in p_str:
        return "boundary_superset"
    if p_str in g_str:
        return "boundary_subset"
    return "boundary_overlap"


def error_taxonomy(predictions: Mapping[str, str], references: Mapping[str, Sequence[str]]) -> dict:
    """``{loại lỗi: số câu}`` cộng tổng số lỗi và số câu."""
    counts = Counter(classify_error(predictions.get(q, ""), golds) for q, golds in references.items())
    n_errors = sum(v for k, v in counts.items() if k is not None)
    return {
        "n": len(references),
        "n_errors": n_errors,
        "counts": {t: counts.get(t, 0) for t in ERROR_TYPES},
        "pct_of_errors": {t: round(100 * counts.get(t, 0) / n_errors, 2) if n_errors else 0.0
                          for t in ERROR_TYPES},
    }


def abstention_stats(predictions: Mapping[str, str], references: Mapping[str, Sequence[str]]) -> dict:
    """"Trả rỗng" như bộ phân loại "câu này không có đáp án"."""
    tp = fp = fn = tn = 0
    for qid, golds in references.items():
        abstained = not tokenize(predictions.get(qid, ""))
        impossible = not golds
        if abstained and impossible:
            tp += 1
        elif abstained:
            fp += 1
        elif impossible:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "abstain_rate": round(100 * (tp + fp) / n, 2) if n else 0.0,
        "true_impossible_rate": round(100 * (tp + fn) / n, 2) if n else 0.0,
        "precision": round(100 * precision, 2),
        "recall": round(100 * recall, 2),
        "f1": round(100 * 2 * precision * recall / (precision + recall), 2) if precision + recall else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def slice_scores(predictions: Mapping[str, str], references: Mapping[str, Sequence[str]],
                 qids: Sequence[str]) -> dict:
    """EM/F1 trung bình trên một tập con câu hỏi."""
    if not qids:
        return {"EM": None, "F1": None, "count": 0}
    em = f1 = 0.0
    for q in qids:
        golds = references[q]
        pred = predictions.get(q, "")
        if golds:
            em += max(exact_match(pred, g) for g in golds)
            f1 += max(token_f1(pred, g) for g in golds)
        else:
            ok = float(not tokenize(pred))
            em += ok
            f1 += ok
    n = len(qids)
    return {"EM": round(100 * em / n, 2), "F1": round(100 * f1 / n, 2), "count": n}


def mcnemar_exact(b: int, c: int) -> float:
    """p-value hai phía của kiểm định McNemar chính xác (nhị thức, p=0,5).

    ``b``, ``c``: số câu chỉ model A đúng / chỉ model B đúng. Câu cả hai cùng đúng
    hoặc cùng sai không mang thông tin về chênh lệch nên không vào phép thử.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_comparison(em_a: Mapping[str, float], em_b: Mapping[str, float],
                      n_boot: int = 2000, seed: int = 42) -> dict:
    """So sánh cặp hai model trên cùng tập câu: bảng 2×2, McNemar, bootstrap CI.

    ``em_a[qid]`` là EM (0/1) của model A trên câu ``qid``.
    """
    qids = sorted(set(em_a) & set(em_b))
    both = only_a = only_b = neither = 0
    diffs = []
    for q in qids:
        a, b = em_a[q] >= 0.5, em_b[q] >= 0.5
        both += a and b
        only_a += a and not b
        only_b += b and not a
        neither += not a and not b
        diffs.append(float(a) - float(b))

    n = len(qids)
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)) if n else []
    lo = boots[int(0.025 * n_boot)] if boots else 0.0
    hi = boots[int(0.975 * n_boot) - 1] if boots else 0.0
    return {
        "n": n,
        "both_correct": both, "only_a": only_a, "only_b": only_b, "neither": neither,
        "em_diff": round(100 * sum(diffs) / n, 2) if n else 0.0,
        "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
        "mcnemar_p": mcnemar_exact(only_a, only_b),
    }
