"""Đo tương thích giữa biên đáp án vàng và biên từ của bộ tách từ.

Đây là công cụ chẩn đoán trung tâm của đồ án (RQ1). Ý tưởng:

Một model word-level như PhoBERT chỉ trả về được dãy TỪ NGUYÊN VẸN (xem
:mod:`mrc.segmented_tokenizer`). Vậy với mỗi câu có đáp án, ta hỏi được — mà
KHÔNG cần chạy model — "biên của đáp án vàng có trùng biên từ không?":

* **aligned**: đáp án bắt đầu và kết thúc đúng tại biên từ. PhoBERT có thể khớp
  chính xác.
* **misaligned**: ít nhất một đầu của đáp án nằm GIỮA một từ đã tách (gold
  "Nội" trong từ "Hà_Nội", hoặc gold "Việt" trong "người_Việt"). Span tốt nhất
  PhoBERT trả về được là span nguyên từ nhỏ nhất bao đáp án — rộng hơn gold.

``oracle_em`` là EM của span nguyên-từ tốt nhất đó: TRẦN EM của một model
word-level trên câu này, dù model hoàn hảo. Câu misaligned có ``oracle_em`` có
thể vẫn là 1 (khi phần thừa chỉ là dấu câu mà chuẩn hoá bỏ đi); ngược lại là 0.

Nhờ tính được trên TOÀN BỘ tập validation (hàng nghìn câu, nhãn do người gán),
phân tích này có cỡ mẫu mà bộ stress-test 50 câu/nhóm không có.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from mrc.metrics import exact_match

__all__ = ["boundary_alignment", "locate_answers"]

Word = tuple[str, int, int]


def locate_answers(context: str, answers: Sequence[str], first_start: int) -> list[tuple[str, int]]:
    """``[(text, start_char)]`` cho mọi đáp án vàng định vị được trong context.

    ``answers[0]`` dùng ``answer_start`` của dataset; các đáp án còn lại (ViQuAD
    có thể có nhiều) được tìm bằng ``find`` vì ``Example`` chỉ lưu offset đầu tiên.
    Đáp án không nằm trong context bị bỏ — không đoán vị trí.
    """
    out: list[tuple[str, int]] = []
    for i, text in enumerate(answers):
        if not text:
            continue
        if i == 0 and first_start >= 0 and context[first_start:first_start + len(text)] == text:
            out.append((text, first_start))
            continue
        found = context.find(text)
        if found >= 0:
            out.append((text, found))
    return out


def _cuts_word(words: Iterable[Word], char: int) -> bool:
    """``char`` nằm GIỮA một từ (không phải tại biên)?"""
    return any(start < char < end for _, start, end in words)


def _enclosing_span(words: Sequence[Word], start: int, end: int) -> tuple[int, int] | None:
    """Span nguyên-từ nhỏ nhất bao ``[start, end)``; ``None`` nếu không từ nào giao."""
    overlapping = [(s, e) for _, s, e in words if s < end and e > start]
    if not overlapping:
        return None
    return min(s for s, _ in overlapping), max(e for _, e in overlapping)


def boundary_alignment(
    context: str, answers: Sequence[str], first_start: int, words: Sequence[Word]
) -> dict | None:
    """Tương thích biên của đáp án vàng với biên từ.

    Returns:
        ``None`` nếu không định vị được đáp án nào (câu impossible, hoặc gold không
        nằm trong context). Ngược lại dict:

        ``aligned``
            ``True`` nếu CÓ ÍT NHẤT MỘT đáp án vàng khớp biên từ ở cả hai đầu.
        ``start_cut`` / ``end_cut``
            Đầu nào của đáp án ĐẦU TIÊN cắt ngang từ.
        ``oracle_em``
            EM tốt nhất một span nguyên-từ đạt được, lấy max trên các đáp án vàng.
        ``oracle_span``
            Chuỗi nguyên-từ tốt nhất tương ứng (để minh hoạ trong báo cáo).
    """
    located = locate_answers(context, answers, first_start)
    if not located:
        return None

    aligned = False
    best_em, best_text = 0.0, ""
    first_cut: tuple[bool, bool] | None = None
    for text, start in located:
        end = start + len(text)
        start_cut, end_cut = _cuts_word(words, start), _cuts_word(words, end)
        if first_cut is None:
            first_cut = (start_cut, end_cut)
        if not (start_cut or end_cut):
            aligned = True

        span = _enclosing_span(words, start, end)
        if span is None:
            continue
        candidate = context[span[0]:span[1]]
        em = max(exact_match(candidate, gold) for gold in answers if gold)
        if em > best_em or not best_text:
            best_em, best_text = em, candidate

    assert first_cut is not None
    return {
        "aligned": aligned,
        "start_cut": first_cut[0],
        "end_cut": first_cut[1],
        "oracle_em": best_em,
        "oracle_span": best_text,
    }
