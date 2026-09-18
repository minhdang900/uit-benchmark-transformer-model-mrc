"""Chọn tokenizer theo loại model, và ghi lại lựa chọn đó cạnh checkpoint.

Hai họ tokenizer trong đồ án khác nhau đúng ở khâu mà đồ án muốn đo:

* **Subword** (XLM-R, SentencePiece): nhận văn bản thô, fast tokenizer trả về
  ``offset_mapping`` sẵn.
* **Word-level** (PhoBERT): nhận văn bản ĐÃ TÁCH TỪ ("Hà_Nội"), và chỉ có slow
  tokenizer — không có ``offset_mapping``. Được bọc bởi
  :class:`mrc.segmented_tokenizer.SegmentedTokenizer`.

Lựa chọn được ghi vào ``mrc_tokenization.json`` trong thư mục checkpoint, để lúc
đánh giá không thể vô tình nạp PhoBERT mà quên bước tách từ — lỗi đó không báo
lỗi, chỉ cho điểm thấp.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "CONFIG_NAME",
    "load_tokenizer",
    "read_tokenization_config",
    "write_tokenization_config",
]

CONFIG_NAME = "mrc_tokenization.json"


def load_tokenizer(model_name: str, word_segmented: bool = False):
    """Tokenizer có ``offset_mapping``, dù model là subword hay word-level."""
    from transformers import AutoTokenizer

    if word_segmented:
        from mrc.segmented_tokenizer import SegmentedTokenizer

        return SegmentedTokenizer(AutoTokenizer.from_pretrained(model_name, use_fast=False))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"{model_name}: không có fast tokenizer ⇒ không có offset_mapping. "
            "Nếu đây là model word-level (PhoBERT), dùng word_segmented=True."
        )
    return tokenizer


def write_tokenization_config(checkpoint: str | Path, **config) -> None:
    path = Path(checkpoint) / CONFIG_NAME
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def read_tokenization_config(checkpoint: str | Path) -> dict:
    """Cấu hình đã ghi khi huấn luyện; ``{}`` nếu checkpoint không có (model ngoài)."""
    path = Path(checkpoint) / CONFIG_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
