"""Tokenizer có ``offset_mapping`` cho model word-level tiếng Việt (PhoBERT).

PhoBERT được pretrain trên văn bản đã TÁCH TỪ: các âm tiết của một từ nối bằng
``_`` ("Hà_Nội", "thủ_đô"). Hai hệ quả cho extractive QA:

1. Phải tách từ TRƯỚC khi tokenize. Ở đây dùng ``pyvi`` (thuần Python). PhoBERT
   gốc dùng RDRSegmenter của VnCoreNLP (cần Java) — hai bộ tách từ không trùng nhau
   hoàn toàn, và đó là một nguồn lệch được nêu trong báo cáo.
2. PhoBERT chỉ có slow tokenizer ⇒ không có ``offset_mapping``. Lớp này tự dựng
   offset: mỗi từ đã tách được căn về khoảng ký tự của nó trong văn bản GỐC, và
   mọi BPE token sinh ra từ từ đó mang chung khoảng ký tự ấy.

Hệ quả của (2) là **tính chất đo được mà đồ án khai thác**: PhoBERT chỉ có thể trả
về đáp án là một dãy TỪ NGUYÊN VẸN theo bộ tách từ. Nếu biên của đáp án vàng cắt
ngang một từ ghép (gold "Nội" trong "Hà_Nội"), không span nào của PhoBERT khớp
chính xác được — EM tối đa của câu đó bằng 0 bất kể model giỏi đến đâu. XLM-R
(subword) không có giới hạn này. Xem :func:`mrc.segmentation.boundary_alignment`.

Lớp này chỉ cài đúng phần giao diện mà :func:`mrc.windowing.make_windows`,
:mod:`mrc.features` và :class:`mrc.transformer_qa.TransformerQA` dùng.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

__all__ = ["SegmentedTokenizer", "pyvi_segment", "segment_with_offsets"]

Word = tuple[str, int, int]  # (từ đã tách, start_char, end_char) trong văn bản GỐC


def pyvi_segment(text: str) -> list[str]:
    """Tách từ bằng pyvi. Trả về danh sách từ, âm tiết trong từ nối bằng ``_``."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        from pyvi import ViTokenizer

    return [w for w in ViTokenizer.tokenize(text).split(" ") if w]


def segment_with_offsets(
    text: str, segment: Callable[[str], list[str]] = pyvi_segment
) -> list[Word]:
    """Tách từ và căn mỗi từ về khoảng ký tự ``[start, end)`` trong ``text`` gốc.

    Bộ tách từ trả về chuỗi đã biến đổi (thêm ``_``, tách dấu câu ra), nên phải
    căn lại. Mỗi âm tiết được tìm tuần tự từ con trỏ hiện tại, bỏ qua khoảng trắng.
    Âm tiết không tìm thấy (bộ tách từ đổi ký tự) bị BỎ QUA thay vì đoán vị trí —
    một offset sai sẽ gán nhãn huấn luyện sai mà không báo lỗi.
    """
    words: list[Word] = []
    pos = 0
    n = len(text)
    for token in segment(text):
        # Chuỗi gốc có thể chứa "_" thật; thử khớp nguyên token trước.
        pieces = [token] if text.startswith(token, _skip_space(text, pos)) else token.split("_")
        start = end = None
        for piece in pieces:
            if not piece:
                continue
            p = _skip_space(text, pos)
            if not text.startswith(piece, p):
                found = text.find(piece, p, min(n, p + 50 + len(piece)))
                if found < 0:
                    continue
                p = found
            if start is None:
                start = p
            end = p + len(piece)
            pos = end
        if start is not None and end is not None:
            words.append((token, start, end))
    return words


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


class _Encoding(dict):
    """Giống ``BatchEncoding`` ở đúng một điểm cần dùng: ``sequence_ids(0)``."""

    def __init__(self, data: dict, seq_ids: list) -> None:
        super().__init__(data)
        self._seq_ids = seq_ids

    def sequence_ids(self, batch_index: int = 0) -> list:
        return self._seq_ids


class SegmentedTokenizer:
    """Bọc slow tokenizer của PhoBERT: tách từ + BPE + ``offset_mapping`` về văn bản gốc."""

    is_fast = False  # trung thực: đây không phải fast tokenizer

    def __init__(
        self,
        tokenizer,
        segment: Callable[[str], list[str]] = pyvi_segment,
        cache_size: int = 200_000,
    ) -> None:
        self._tok = tokenizer
        self._segment = segment
        self._cache: OrderedDict[str, tuple[list[int], list[tuple[int, int]]]] = OrderedDict()
        self._cache_size = cache_size

        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    # ── phần giao diện mà pipeline dùng ─────────────────────────────────────
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._tok.num_special_tokens_to_add(pair=pair)

    def save_pretrained(self, path) -> None:
        self._tok.save_pretrained(path)

    def words(self, text: str) -> list[Word]:
        """Các từ đã tách kèm offset — dùng cho phân tích biên từ."""
        return segment_with_offsets(text, self._segment)

    def encode_text(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """``(input_ids, offsets)`` của văn bản, KHÔNG token đặc biệt. Có cache."""
        hit = self._cache.get(text)
        if hit is not None:
            self._cache.move_to_end(text)
            return hit

        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for word, start, end in segment_with_offsets(text, self._segment):
            pieces = self._tok.tokenize(word)
            if not pieces:
                continue
            ids.extend(self._tok.convert_tokens_to_ids(pieces))
            # Mọi BPE piece của một từ mang chung khoảng ký tự của cả từ: model
            # word-level không thể trả về nửa từ.
            offsets.extend([(start, end)] * len(pieces))

        result = (ids, offsets)
        self._cache[text] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    def __call__(
        self,
        text: str,
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        truncation: bool | str = False,
        max_length: int | None = None,
        padding: bool | str = False,
        **_ignored,
    ) -> _Encoding:
        ids_a, off_a = self.encode_text(text)

        if text_pair is None:
            if add_special_tokens:
                ids = [self.cls_token_id, *ids_a, self.sep_token_id]
                offsets = [(0, 0), *off_a, (0, 0)]
                seq = [None, *([0] * len(ids_a)), None]
            else:
                ids, offsets, seq = list(ids_a), list(off_a), [0] * len(ids_a)
        else:
            ids_b, off_b = self.encode_text(text_pair)
            if max_length is not None and truncation in (True, "only_second"):
                room = max_length - len(ids_a) - self.num_special_tokens_to_add(pair=True)
                if room < 0:
                    raise ValueError(f"Câu hỏi dài {len(ids_a)} token vượt max_length={max_length}")
                ids_b, off_b = ids_b[:room], off_b[:room]
            # RoBERTa: <s> A </s></s> B </s>
            ids = [self.cls_token_id, *ids_a, self.sep_token_id, self.sep_token_id,
                   *ids_b, self.sep_token_id]
            offsets = [(0, 0), *([(0, 0)] * len(ids_a)), (0, 0), (0, 0), *off_b, (0, 0)]
            seq = [None, *([0] * len(ids_a)), None, None, *([1] * len(ids_b)), None]

        data = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            data["offset_mapping"] = offsets
        return _Encoding(data, seq)
