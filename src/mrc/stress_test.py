"""Nạp và KIỂM ĐỊNH bộ stress-test 250 câu (5 nhóm E1–E5 × 50).

Kiểm định chạy TRƯỚC khi bộ này được dùng để chấm bất kỳ model nào. Lý do: một
câu có đáp án mà đáp án vàng không nằm trong context thì không model extractive
nào trả lời đúng được — chấm trên câu đó đo lỗi của DỮ LIỆU, không phải của model.

Mỗi tệp nhóm chứa hai loại bản ghi:

* **SQuAD** — ``{"title", "paragraphs": [{"context", "qas": [...]}]}``: 50 câu/nhóm,
  là 250 câu mà tài liệu dự án mô tả.
* **Phẳng** — ``{"id", "context", "question", "expected_answer", ...}``: 24/nhóm,
  không thuộc lược đồ SQuAD, không được mô tả ở đâu. Bị kiểm định nhưng KHÔNG chấm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mrc.data import Example

__all__ = ["CATEGORIES", "audit_item", "load_stress_test"]

#: Mã nhóm → tên tệp, theo thứ tự trong tài liệu dự án.
CATEGORIES = {
    "E1": "compound_word_boundary",
    "E2": "distractor_contexts",
    "E3": "negative_questions",
    "E4": "multi_hop_reasoning",
    "E5": "ambiguous_ground_truth",
}


def _template(question: str) -> str:
    """Khung câu hỏi sau khi thay phần trong nháy đơn bằng X — để đếm câu hỏi mẫu."""
    return re.sub(r"'[^']*'", "'X'", question)


def audit_item(context: str, qa: dict) -> dict:
    """Các kiểm tra tự động cho một câu hỏi dạng SQuAD."""
    texts = [t for t in qa.get("answers", {}).get("text", []) if t]
    starts = qa.get("answers", {}).get("answer_start", [])
    answerable = bool(texts) and not qa.get("is_impossible", False)
    in_context = all(t in context for t in texts) if answerable else None
    offset_ok = (
        all(context[s:s + len(t)] == t for t, s in zip(texts, starts)) if answerable else None
    )
    in_question = any(t.lower() in qa["question"].lower() for t in texts) if answerable else None
    return {
        "answerable": answerable,
        "answer_in_context": in_context,
        "offset_correct": offset_ok,
        "answer_in_question": in_question,
        "template": _template(qa["question"]),
        # Câu impossible luôn chấm được (đáp án đúng là rỗng); câu có đáp án chỉ
        # chấm được khi đáp án nằm trong context.
        "valid": (not answerable) or bool(in_context),
    }


def load_stress_test(directory: str | Path) -> tuple[list[Example], dict[str, dict], dict]:
    """``(examples, meta_by_qid, file_stats)``.

    ``meta_by_qid[qid]`` = ``{"category": "E1", **audit_item(...)}``. Offset sai
    nhưng đáp án có trong context được sửa bằng ``find`` — offset chỉ dùng cho
    phân tích biên từ, còn EM/F1 chấm bằng chuỗi.
    """
    directory = Path(directory)
    examples: list[Example] = []
    meta: dict[str, dict] = {}
    stats: dict[str, dict] = {}

    for code, name in CATEGORIES.items():
        records = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))["data"]
        squad = [r for r in records if "paragraphs" in r]
        flat = [r for r in records if "paragraphs" not in r]
        stats[code] = {
            "file": f"{name}.json",
            "squad_questions": sum(len(p["qas"]) for r in squad for p in r["paragraphs"]),
            "flat_records": len(flat),
            "flat_answer_in_context": sum(r.get("expected_answer", "") in r.get("context", "")
                                          for r in flat),
        }
        for article in squad:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    audit = audit_item(context, qa)
                    texts = [t for t in qa["answers"]["text"] if t] if audit["answerable"] else []
                    start = context.find(texts[0]) if texts and texts[0] in context else -1
                    examples.append(Example(
                        qid=qa["id"], question=qa["question"], context=context,
                        title=article.get("title", ""), answers=texts,
                        answer_start=start, is_impossible=not audit["answerable"],
                    ))
                    meta[qa["id"]] = {"category": code, **audit}
    return examples, meta, stats
