"""Bất biến của tokenizer word-level và phép đo biên từ.

Test dùng bộ tách từ và BPE GIẢ (xác định, không cần tải model) cho các bất biến
cấu trúc, và pyvi THẬT cho phép căn offset trên câu tiếng Việt.
"""
from __future__ import annotations

import pytest

from mrc.data import Example
from mrc.features import prepare_train_features
from mrc.segmentation import boundary_alignment, locate_answers
from mrc.segmented_tokenizer import SegmentedTokenizer, segment_with_offsets
from mrc.windowing import make_windows

COMPOUNDS = {"Hà Nội": "Hà_Nội", "thủ đô": "thủ_đô", "Việt Nam": "Việt_Nam"}


def fake_segment(text: str) -> list[str]:
    """Nối các từ ghép đã biết bằng '_' và tách dấu câu — giống pyvi ở hình dạng."""
    for plain, joined in COMPOUNDS.items():
        text = text.replace(plain, joined)
    for p in ",.()":
        text = text.replace(p, f" {p} ")
    return [w for w in text.split(" ") if w]


class FakeBPE:
    """Mỗi từ -> 1 id; từ dài hơn 6 ký tự -> 2 piece, để kiểm offset dùng chung."""

    cls_token_id, pad_token_id, sep_token_id = 0, 1, 2

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def tokenize(self, word: str) -> list[str]:
        return [word[:3] + "@@", word[3:]] if len(word) > 6 else [word]

    def convert_tokens_to_ids(self, pieces):
        return [self.vocab.setdefault(p, len(self.vocab) + 10) for p in pieces]

    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2

    def save_pretrained(self, path):
        pass


@pytest.fixture
def tok():
    return SegmentedTokenizer(FakeBPE(), segment=fake_segment)


CONTEXT = "Hà Nội là thủ đô của Việt Nam, thành phố lớn."


# ── căn offset ───────────────────────────────────────────────────────────────
def test_offsets_slice_back_to_original_words():
    words = segment_with_offsets(CONTEXT, fake_segment)
    for word, start, end in words:
        assert CONTEXT[start:end] == word.replace("_", " ")
    assert [w for w, _, _ in words][:3] == ["Hà_Nội", "là", "thủ_đô"]


def test_punctuation_gets_its_own_offset():
    words = segment_with_offsets(CONTEXT, fake_segment)
    comma = [w for w in words if w[0] == ","][0]
    assert CONTEXT[comma[1]:comma[2]] == ","


def test_real_pyvi_alignment_reconstructs_text():
    text = "Nguyễn Tất Thành rời bến Nhà Rồng năm 1911, (khi ông 21 tuổi)."
    words = segment_with_offsets(text)
    assert words, "pyvi không trả về từ nào"
    for word, start, end in words:
        assert text[start:end] == word.replace("_", " ")
    # không chồng lấp, tăng dần
    for (_, _, e1), (_, s2, _) in zip(words, words[1:]):
        assert e1 <= s2


# ── mã hoá ───────────────────────────────────────────────────────────────────
def test_bpe_pieces_share_the_word_span(tok):
    ids, offsets = tok.encode_text("khánh_thành")  # 1 từ dài -> 2 piece
    assert len(ids) == 2
    assert offsets[0] == offsets[1]


def test_pair_format_and_sequence_ids(tok):
    enc = tok("Thủ đô?", "Hà Nội là thủ đô.", return_offsets_mapping=True)
    ids, seq = enc["input_ids"], enc.sequence_ids(0)
    assert ids[0] == 0 and ids[-1] == 2
    assert len(ids) == len(seq) == len(enc["offset_mapping"]) == len(enc["attention_mask"])
    first_ctx = seq.index(1)
    assert ids[first_ctx - 2:first_ctx] == [2, 2]  # </s></s> giữa hai đoạn


def test_truncation_only_cuts_the_context(tok):
    long_ctx = " ".join(["từ"] * 100)
    enc = tok("câu hỏi", long_ctx, truncation="only_second", max_length=20,
              return_offsets_mapping=True)
    assert len(enc["input_ids"]) == 20
    assert enc.sequence_ids(0).count(0) == 2  # câu hỏi giữ nguyên


# ── tích hợp với pipeline ────────────────────────────────────────────────────
def test_windows_decode_to_whole_words_of_the_original(tok):
    ctx = " ".join(f"câu {i} nói về Hà Nội." for i in range(30))
    windows = make_windows("Hà Nội ở đâu?", ctx, tok, max_length=40, doc_stride=10)
    assert len(windows) > 1
    for w in windows:
        for off in w.offset_mapping:
            if off is not None:
                piece = ctx[off[0]:off[1]]
                assert piece and piece == piece.strip()


def test_aligned_answer_labels_decode_to_gold(tok):
    ex = Example(qid="a", question="Thủ đô là gì?", context=CONTEXT,
                 answers=["Hà Nội"], answer_start=0)
    f = prepare_train_features([ex], tok, max_length=64, doc_stride=16)
    s, e = f["start_positions"][0], f["end_positions"][0]
    assert s > 0
    enc = make_windows(ex.question, ex.context, tok, max_length=64, doc_stride=16)[0]
    decoded = CONTEXT[enc.offset_mapping[s][0]:enc.offset_mapping[e][1]]
    assert decoded == "Hà Nội"


def test_misaligned_answer_is_widened_to_enclosing_word(tok):
    """Gold 'Nội' nằm GIỮA từ 'Hà_Nội': nhãn tốt nhất có được là cả từ."""
    start = CONTEXT.index("Nội")
    ex = Example(qid="m", question="?", context=CONTEXT, answers=["Nội"], answer_start=start)
    f = prepare_train_features([ex], tok, max_length=64, doc_stride=16)
    s, e = f["start_positions"][0], f["end_positions"][0]
    enc = make_windows(ex.question, ex.context, tok, max_length=64, doc_stride=16)[0]
    assert CONTEXT[enc.offset_mapping[s][0]:enc.offset_mapping[e][1]] == "Hà Nội"


# ── phép đo biên từ ──────────────────────────────────────────────────────────
WORDS = segment_with_offsets(CONTEXT, fake_segment)


def test_aligned_answer_has_oracle_em_one():
    r = boundary_alignment(CONTEXT, ["thủ đô"], CONTEXT.index("thủ đô"), WORDS)
    assert r["aligned"] and r["oracle_em"] == 1.0


def test_answer_cutting_a_compound_is_misaligned_with_oracle_zero():
    r = boundary_alignment(CONTEXT, ["Nội"], CONTEXT.index("Nội"), WORDS)
    assert not r["aligned"]
    assert r["start_cut"] and not r["end_cut"]
    assert r["oracle_em"] == 0.0 and r["oracle_span"] == "Hà Nội"


def test_any_aligned_gold_makes_item_aligned():
    r = boundary_alignment(CONTEXT, ["Nội", "Hà Nội"], CONTEXT.index("Nội"), WORDS)
    assert r["aligned"] and r["oracle_em"] == 1.0


def test_unlocatable_answer_returns_none():
    assert boundary_alignment(CONTEXT, ["Paris"], -1, WORDS) is None
    assert boundary_alignment(CONTEXT, [], -1, WORDS) is None


def test_locate_answers_prefers_dataset_offset_then_find():
    ctx = "Nam và Nam"
    assert locate_answers(ctx, ["Nam"], 7) == [("Nam", 7)]
    assert locate_answers(ctx, ["Nam"], 3) == [("Nam", 0)]  # offset sai -> find
