"""Bộ stress-test v2: dựng lại từ UIT-ViQuAD 2.0 validation, kiểm định được.

Bộ v1 250 câu (đã gỡ khỏi kho mã) không dùng được làm thước đo: 105/120 câu có đáp
án mà đáp án vàng không nằm trong context, 5 context cho 250 câu, câu hỏi sinh bằng
template. Bộ v2 thay đổi nguyên tắc:

1. **Không tự viết câu hỏi hay tự gán nhãn.** Mọi câu hỏi và đáp án vàng lấy từ
   validation (do người gán). Có hai loại mục:

   * *slice* — câu gốc, giữ nguyên, được CHỌN vì có một hiện tượng đo được;
   * *perturbation* — câu gốc bị biến đổi bằng một luật mà nhãn mới SUY RA được
     từ nhãn cũ (xoá câu chứa đáp án ⇒ không trả lời được; thêm câu nhiễu không
     nói về chủ thể của câu hỏi ⇒ đáp án không đổi; bỏ dấu câu hỏi ⇒ đáp án không
     đổi). Mỗi perturbation đi kèm câu gốc (``role="original"``) để đo độ nhất quán
     theo cặp, không chỉ điểm trung bình.

2. **Mỗi mục đều chấm được**, và bất biến đó được kiểm bằng :func:`audit`.

3. **Không trộn vào huấn luyện.** Context lấy từ validation, vốn không giao với
   train (0/557 context). Bộ này chỉ dùng để chẩn đoán sau khi mô hình đã chốt;
   không chọn epoch, seed hay ngưỡng τ trên nó.

Năm nhóm:

====  =================================  =============================================
Mã    Hiện tượng                         Cách dựng
====  =================================  =============================================
E1    Ranh giới từ ghép                  slice: biên đáp án cắt ngang từ pyvi (E1a);
                                         biên đáp án là từ ghép riêng nhiều âm tiết (E1b)
E2    Nhiễu / ngữ cảnh dài               slice: đáp án nằm sau âm tiết thứ 180 (E2a);
                                         đáp án số, context có ≥3 số khác (E2b);
                                         perturbation: thêm câu nhiễu (E2c, có cặp)
E3    Không trả lời được                 slice: câu impossible gốc (E3a);
                                         perturbation: xoá câu chứa đáp án (E3b, có cặp)
E4    Lệch từ vựng / nhiều câu           slice: câu có đáp án KHÔNG phải câu trùng từ
                                         với câu hỏi nhiều nhất
E5    Câu hỏi không dấu                  perturbation: bỏ dấu tiếng Việt ở câu hỏi (có cặp)
====  =================================  =============================================
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mrc.data import Example
from mrc.normalize import normalize_answer, tokenize

__all__ = [
    "CATEGORIES",
    "SUBSETS",
    "StressItem",
    "sentence_spans",
    "strip_diacritics",
    "remove_answer_sentence",
    "make_distractor",
    "lexical_gap",
    "build",
    "audit",
    "to_squad",
    "load_stress_v2",
]

CATEGORIES = {
    "E1": "Ranh giới từ ghép",
    "E2": "Nhiễu / ngữ cảnh dài",
    "E3": "Không trả lời được",
    "E4": "Lệch từ vựng / nhiều câu",
    "E5": "Câu hỏi không dấu",
}

#: Mô tả từng tập con; khoá là mã tập con ghi trong ``meta.subset``.
SUBSETS = {
    "E1a": "slice — biên đáp án vàng cắt ngang một từ của pyvi",
    "E1b": "slice — đáp án mở đầu/kết thúc bằng từ ghép riêng (viết hoa) ≥2 âm tiết",
    "E2a": "slice — đáp án nằm sau âm tiết thứ 180 của context (ngoài cửa sổ đầu)",
    "E2b": "slice — đáp án chứa số, context có ≥3 con số khác làm ứng viên nhiễu",
    "E2c": "perturbation — chèn một câu nhiễu nói về chủ thể khác, số khác",
    "E3a": "slice — câu không trả lời được do người viết (ViQuAD 2.0)",
    "E3b": "perturbation — xoá câu chứa đáp án khỏi context",
    "E4": "slice — câu chứa đáp án không phải câu trùng từ nhiều nhất với câu hỏi",
    "E5": "perturbation — bỏ toàn bộ dấu tiếng Việt trong câu hỏi",
}

#: Số mục mặc định cho mỗi tập con (trước khi thêm câu gốc đi cặp).
DEFAULT_SIZES = {
    "E1a": None,  # lấy tất cả — hiện tượng hiếm (≈1,2% câu có đáp án)
    "E1b": 100,
    "E2a": 80,
    "E2b": 80,
    "E2c": 100,
    "E3a": 100,
    "E3b": 100,
    "E4": 120,
    "E5": 150,
}

#: Tối đa bao nhiêu mục của cùng một tập con được lấy từ một context, để bộ dữ
#: liệu trải trên nhiều đoạn văn thay vì dồn vào vài đoạn (lỗi của bộ v1).
MAX_PER_CONTEXT = 2

#: Ngưỡng vị trí (âm tiết) của E2a. PhoBERT dùng cửa sổ 256 token; 180 âm tiết
#: đầu gần như luôn nằm trọn trong cửa sổ đầu, nên đáp án sau đó buộc mô hình
#: phải xử lý đúng cửa sổ thứ hai trở đi.
LONG_CONTEXT_OFFSET = 180


@dataclass
class StressItem:
    """Một câu của bộ stress-test, ở dạng đã sẵn sàng ghi ra SQuAD."""

    qid: str
    question: str
    context: str
    title: str
    answers: list[str]
    answer_starts: list[int]
    category: str
    subset: str
    role: str  # "slice" | "perturbed" | "original"
    source_qid: str
    pair_id: str | None = None
    note: str = ""

    @property
    def is_impossible(self) -> bool:
        return not self.answers


# ── tiện ích văn bản ─────────────────────────────────────────────────────────────

_SENT_END = re.compile(r"(?<=[.!?…])[\"”’)\]]*\s+(?=[\"“‘(\[]?[A-ZÀ-Ỹ0-9Đ])")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """``[(start, end)]`` của từng câu trong ``text`` — offset ký tự, không mất ký tự.

    Tách tại dấu kết thúc câu theo sau bởi khoảng trắng và một chữ hoa/chữ số.
    Không tách tại "TP.HCM", "3.5" hay "v.v." giữa câu vì các trường hợp đó không
    có khoảng trắng hoặc không có chữ hoa theo sau.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.start()
        # m bắt đầu ngay sau dấu câu; phần dấu nháy/ngoặc đóng thuộc câu trước.
        close = re.match(r"[\"”’)\]]*", text[end:]).end()
        end += close
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans


def _sentence_index(spans: Sequence[tuple[int, int]], char: int) -> int | None:
    for i, (s, e) in enumerate(spans):
        if s <= char < e:
            return i
    return None


def strip_diacritics(text: str) -> str:
    """Bỏ mọi dấu tiếng Việt (thanh và mũ/móc), ``đ`` → ``d``. Giữ hoa/thường."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    )


def _locate(context: str, answers: Sequence[str], first_start: int) -> list[int]:
    """Offset của từng đáp án; ``-1`` nếu không tìm thấy."""
    out = []
    for i, a in enumerate(answers):
        if i == 0 and first_start >= 0 and context[first_start:first_start + len(a)] == a:
            out.append(first_start)
        else:
            out.append(context.find(a))
    return out


# ── perturbation ─────────────────────────────────────────────────────────────────


def remove_answer_sentence(ex: Example) -> tuple[str, str] | None:
    """Xoá câu chứa đáp án. Trả ``(context_mới, câu_đã_xoá)`` hoặc ``None``.

    Chỉ nhận khi nhãn mới chắc chắn là "không trả lời được" theo các kiểm tra
    tự động được:

    * đáp án vàng nằm gọn trong MỘT câu;
    * sau khi xoá, không đáp án vàng nào (kể cả sau chuẩn hoá) còn trong context;
    * context còn lại ≥2 câu, để câu hỏi vẫn "có vẻ" trả lời được.

    Diễn đạt lại đáp án ở câu khác (đồng nghĩa) không kiểm tự động được — các mục
    này được đưa vào tệp kiểm tra tay.
    """
    if ex.is_impossible or not ex.answers or ex.answer_start < 0:
        return None
    spans = sentence_spans(ex.context)
    if len(spans) < 3:
        return None
    end_char = ex.answer_start + len(ex.answers[0]) - 1
    i, j = _sentence_index(spans, ex.answer_start), _sentence_index(spans, end_char)
    if i is None or i != j:
        return None
    s, e = spans[i]
    # Xoá cả khoảng trắng theo sau để không để lại hai dấu cách liền nhau.
    tail = e
    while tail < len(ex.context) and ex.context[tail].isspace():
        tail += 1
    new = (ex.context[:s] + ex.context[tail:]).strip()
    norm_new = normalize_answer(new)
    for a in ex.answers:
        if a in new or (normalize_answer(a) and f" {normalize_answer(a)} " in f" {norm_new} "):
            return None
    return new, ex.context[s:e]


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_NUMBER_WITH_UNIT = re.compile(r"(?:\b((?i:ngày|tháng|thế kỷ|thế kỉ))\s+)?(\d+(?:[.,]\d+)*)")

#: Loại thực thể của các tên bài trong ViQuAD validation, và vài tên thay thế cùng
#: loại không có trong validation. Câu nhiễu chỉ thay chủ thể bằng một thực thể
#: CÙNG LOẠI (địa danh ↔ địa danh, người ↔ người) để câu vẫn tự nhiên; tên bài
#: không thuộc hai loại này (Kiến, Thần học, Nước biển...) không được dùng.
ENTITY_TYPES = {
    "place": ["Paris", "Nouvelle-Calédonie", "Canada", "Puerto Rico", "Nam Ossetia",
              "Mông Cổ", "Montréal", "Uganda", "Texas",
              "Brasil", "Ai Cập", "Na Uy", "Chile", "Ontario", "Kenya", "Bavaria"],
    "person": ["François Mitterrand", "Mahatma Gandhi", "Franklin D. Roosevelt",
               "Winston Churchill", "Charles de Gaulle", "Jawaharlal Nehru"],
}


def _shift_number(token: str, rng: random.Random, unit: str | None = None) -> str:
    """Một con số khác, cùng hình dạng và vẫn hợp lệ.

    ``unit`` là từ đứng trước để số mới vẫn hợp lệ: ngày 1–28, tháng 1–12,
    thế kỷ 1–21; năm (4 chữ số, 1000–2100) lệch 3–40 năm và không vượt quá 2024;
    số khác đổi từng chữ số.
    """
    unit = unit.lower().replace("kỉ", "kỷ") if unit else None
    limits = {"ngày": 28, "tháng": 12, "thế kỷ": 21}
    if unit in limits and token.isdigit():
        hi = limits[unit]
        choices = [str(v) for v in range(1, hi + 1) if str(v) != token]
        return rng.choice(choices)
    digits = re.sub(r"\D", "", token)
    if len(digits) == 4 and 1000 <= int(digits) <= 2100 and token == digits:  # năm
        delta = rng.randint(3, 40)
        year = int(digits)
        return str(year - delta if year + delta > 2024 or rng.random() < 0.5 else year + delta)
    out = []
    for ch in token:
        out.append(str((int(ch) + rng.randint(1, 8)) % 10) if ch.isdigit() else ch)
    s = "".join(out)
    if s[0] == "0" and len(re.sub(r"\D", "", s)) > 1:
        s = str(rng.randint(1, 9)) + s[1:]
    return s if s != token else _shift_number(token, rng)


def _shift_all_numbers(text: str, rng: random.Random) -> str:
    def repl(m: re.Match) -> str:
        unit, num = m.group(1), m.group(2)
        new = _shift_number(num, rng, unit)
        return f"{unit} {new}" if unit else new
    return _NUMBER_WITH_UNIT.sub(repl, text)


def make_distractor(
    ex: Example, rng: random.Random, entity_types: dict[str, list[str]] | None = None
) -> tuple[str, str, int] | None:
    """Câu nhiễu kiểu AddSent (Jia & Liang, 2017) bằng luật. Trả
    ``(context_mới, câu_nhiễu, offset_đáp_án_mới)`` hoặc ``None``.

    Lấy câu chứa đáp án, thay chủ thể (tên bài — xuất hiện ở cả câu hỏi và câu
    đáp án) bằng một thực thể khác cùng loại, và thay mọi con số bằng số khác.
    Câu nhiễu giống câu hỏi về mặt từ vựng nhưng nói về chủ thể khác, nên đáp án
    đúng không đổi. Chỉ áp dụng cho đáp án có chữ số, để câu nhiễu mang một
    "đáp án giả" cùng loại.
    """
    entity_types = entity_types or ENTITY_TYPES
    if ex.is_impossible or not ex.title or ex.answer_start < 0:
        return None
    kind = next((k for k, names in entity_types.items() if ex.title in names), None)
    if kind is None:
        return None
    gold = ex.answers[0]
    if not _NUMBER.search(gold) or ex.title in gold or ex.title not in ex.question:
        return None
    spans = sentence_spans(ex.context)
    i = _sentence_index(spans, ex.answer_start)
    if i is None:
        return None
    s, e = spans[i]
    sentence = ex.context[s:e]
    if ex.title not in sentence:
        return None
    candidates = [t for t in entity_types[kind] if t != ex.title and t not in ex.context
                  and ex.title not in t and t not in ex.title]
    if not candidates:
        return None
    fake = sentence.replace(ex.title, rng.choice(candidates))
    fake = _shift_all_numbers(fake, rng)
    if any(a in fake for a in ex.answers) or fake == sentence:
        return None
    if rng.random() < 0.5:
        new_ctx = ex.context.rstrip() + " " + fake
        new_start = ex.answer_start
    else:
        new_ctx = fake + " " + ex.context
        new_start = ex.answer_start + len(fake) + 1
    if new_ctx[new_start:new_start + len(gold)] != gold:
        return None
    return new_ctx, fake, new_start


def lexical_gap(ex: Example) -> bool:
    """Câu chứa đáp án KHÔNG phải câu có nhiều âm tiết chung với câu hỏi nhất.

    Mô hình chỉ khớp từ bề mặt sẽ chọn nhầm câu; trả lời đúng cần hiểu đồng nghĩa,
    đồng tham chiếu, hoặc ghép thông tin từ câu khác. Đây là xấp xỉ — không phân
    biệt được suy luận nhiều bước thật với diễn đạt lại.
    """
    if ex.is_impossible or ex.answer_start < 0:
        return False
    spans = sentence_spans(ex.context)
    if len(spans) < 2:
        return False
    i = _sentence_index(spans, ex.answer_start)
    if i is None:
        return False
    q = set(tokenize(ex.question))
    overlap = [len(q & set(tokenize(ex.context[s:e]))) for s, e in spans]
    return max(overlap) > overlap[i]


# ── dựng bộ dữ liệu ──────────────────────────────────────────────────────────────


def _capped_sample(pool: Sequence[Example], n: int | None, rng: random.Random) -> list[Example]:
    """Lấy mẫu ngẫu nhiên có seed, tối đa ``MAX_PER_CONTEXT`` câu mỗi context."""
    pool = list(pool)
    rng.shuffle(pool)
    per_ctx: Counter = Counter()
    out = []
    for ex in pool:
        if per_ctx[ex.context] >= MAX_PER_CONTEXT:
            continue
        per_ctx[ex.context] += 1
        out.append(ex)
        if n is not None and len(out) >= n:
            break
    return out


def _slice(ex: Example, subset: str, k: int, note: str = "") -> StressItem:
    starts = _locate(ex.context, ex.answers, ex.answer_start) if ex.answers else []
    return StressItem(
        qid=f"s2-{subset}-{k:04d}", question=ex.question, context=ex.context, title=ex.title,
        answers=list(ex.answers), answer_starts=starts, category=subset[:2], subset=subset,
        role="slice", source_qid=ex.qid, note=note,
    )


def _original_twin(ex: Example, subset: str, k: int) -> StressItem:
    item = _slice(ex, subset, k)
    item.qid = f"s2-{subset}-{k:04d}-orig"
    item.role = "original"
    item.pair_id = f"s2-{subset}-{k:04d}"
    return item


def build(
    examples: Sequence[Example],
    words_by_context: dict[str, list] | None = None,
    sizes: dict[str, int | None] | None = None,
    seed: int = 42,
    segment: Callable | None = None,
) -> list[StressItem]:
    """Dựng toàn bộ bộ stress-test v2 từ ``examples`` (validation).

    ``words_by_context`` là kết quả :func:`mrc.segmented_tokenizer.segment_with_offsets`
    cho từng context; nếu ``None`` sẽ tự tính bằng ``segment`` (mặc định pyvi).
    Mọi bước lấy mẫu dùng một ``Random`` riêng theo tập con, nên đổi kích thước một
    tập con không làm thay đổi các tập con khác.
    """
    from mrc.segmentation import boundary_alignment

    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    answerable = [e for e in examples if not e.is_impossible and e.answers and e.answer_start >= 0
                  and e.context[e.answer_start:e.answer_start + len(e.answers[0])] == e.answers[0]]
    if words_by_context is None:
        from mrc.segmented_tokenizer import segment_with_offsets

        seg = segment or segment_with_offsets
        words_by_context = {c: seg(c) for c in {e.context for e in answerable}}

    def rng_for(subset: str) -> random.Random:
        return random.Random(f"{seed}-{subset}")

    items: list[StressItem] = []

    # E1 ─ ranh giới từ ghép
    e1a, e1b = [], []
    for ex in answerable:
        words = words_by_context[ex.context]
        r = boundary_alignment(ex.context, ex.answers, ex.answer_start, words)
        if r is None:
            continue
        if not r["aligned"]:
            e1a.append(ex)
            continue
        end = ex.answer_start + len(ex.answers[0])
        inside = [w for w in words if w[1] < end and w[2] > ex.answer_start]
        edge = [inside[0], inside[-1]] if inside else []
        if any("_" in w[0] and w[0][:1].isupper() for w in edge):
            e1b.append(ex)
    for k, ex in enumerate(_capped_sample(e1a, sizes["E1a"], rng_for("E1a"))):
        items.append(_slice(ex, "E1a", k))
    for k, ex in enumerate(_capped_sample(e1b, sizes["E1b"], rng_for("E1b"))):
        items.append(_slice(ex, "E1b", k))

    # E2 ─ nhiễu / ngữ cảnh dài
    e2a = [e for e in answerable if len(e.context[:e.answer_start].split()) > LONG_CONTEXT_OFFSET]
    e2b = []
    for ex in answerable:
        if not _NUMBER.search(ex.answers[0]):
            continue
        others = set(_NUMBER.findall(ex.context)) - set(_NUMBER.findall(" ".join(ex.answers)))
        if len(others) >= 3:
            e2b.append(ex)
    for k, ex in enumerate(_capped_sample(e2a, sizes["E2a"], rng_for("E2a"))):
        items.append(_slice(ex, "E2a", k))
    for k, ex in enumerate(_capped_sample(e2b, sizes["E2b"], rng_for("E2b"))):
        items.append(_slice(ex, "E2b", k))

    rng = rng_for("E2c")
    k = 0
    for ex in _capped_sample(answerable, None, rng):
        if sizes["E2c"] is not None and k >= sizes["E2c"]:
            break
        made = make_distractor(ex, rng)
        if made is None:
            continue
        new_ctx, fake, new_start = made
        starts = [new_start] + [new_ctx.find(a) for a in ex.answers[1:]]
        items.append(StressItem(
            qid=f"s2-E2c-{k:04d}", question=ex.question, context=new_ctx, title=ex.title,
            answers=list(ex.answers), answer_starts=starts, category="E2", subset="E2c",
            role="perturbed", source_qid=ex.qid, pair_id=f"s2-E2c-{k:04d}",
            note=f"câu nhiễu: {fake}",
        ))
        items.append(_original_twin(ex, "E2c", k))
        k += 1

    # E3 ─ không trả lời được
    impossible = [e for e in examples if e.is_impossible]
    for k, ex in enumerate(_capped_sample(impossible, sizes["E3a"], rng_for("E3a"))):
        items.append(_slice(ex, "E3a", k))
    rng = rng_for("E3b")
    k = 0
    for ex in _capped_sample(answerable, None, rng):
        if sizes["E3b"] is not None and k >= sizes["E3b"]:
            break
        made = remove_answer_sentence(ex)
        if made is None:
            continue
        new_ctx, removed = made
        items.append(StressItem(
            qid=f"s2-E3b-{k:04d}", question=ex.question, context=new_ctx, title=ex.title,
            answers=[], answer_starts=[], category="E3", subset="E3b", role="perturbed",
            source_qid=ex.qid, pair_id=f"s2-E3b-{k:04d}", note=f"câu đã xoá: {removed}",
        ))
        items.append(_original_twin(ex, "E3b", k))
        k += 1

    # E4 ─ lệch từ vựng
    e4 = [e for e in answerable if lexical_gap(e)]
    for k, ex in enumerate(_capped_sample(e4, sizes["E4"], rng_for("E4"))):
        items.append(_slice(ex, "E4", k))

    # E5 ─ câu hỏi không dấu (giữ tỉ lệ có đáp án / không như validation)
    rng = rng_for("E5")
    ok_ids = {e.qid for e in answerable}
    pool = [e for e in examples if strip_diacritics(e.question) != e.question
            and (e.is_impossible or e.qid in ok_ids)]
    for k, ex in enumerate(_capped_sample(pool, sizes["E5"], rng)):
        starts = _locate(ex.context, ex.answers, ex.answer_start) if ex.answers else []
        items.append(StressItem(
            qid=f"s2-E5-{k:04d}", question=strip_diacritics(ex.question), context=ex.context,
            title=ex.title, answers=list(ex.answers), answer_starts=starts, category="E5",
            subset="E5", role="perturbed", source_qid=ex.qid, pair_id=f"s2-E5-{k:04d}",
            note=f"câu hỏi gốc: {ex.question}",
        ))
        items.append(_original_twin(ex, "E5", k))

    ids = [it.qid for it in items]
    assert len(ids) == len(set(ids)), "qid trùng"
    return items


# ── kiểm định ────────────────────────────────────────────────────────────────────


def audit(items: Sequence[StressItem], train_contexts: set[str] | None = None) -> dict:
    """Các bất biến mà mọi mục phải thoả; trả về thống kê + danh sách vi phạm.

    Một bộ stress-test hợp lệ khi ``violations`` rỗng.
    """
    violations: list[dict] = []

    def bad(it: StressItem, why: str) -> None:
        violations.append({"qid": it.qid, "reason": why})

    for it in items:
        if it.answers:
            if len(it.answer_starts) != len(it.answers):
                bad(it, "số offset khác số đáp án")
            for a, s in zip(it.answers, it.answer_starts):
                if s < 0 or it.context[s:s + len(a)] != a:
                    bad(it, f"offset sai hoặc đáp án không nằm trong context: {a!r}")
        if it.subset == "E3b" and it.role == "perturbed" and not it.note.startswith("câu đã xoá: "):
            bad(it, "thiếu ghi chú câu đã xoá")
        if it.subset == "E5" and it.role == "perturbed":
            if strip_diacritics(it.question) != it.question:
                bad(it, "câu hỏi E5 vẫn còn dấu")
        if train_contexts is not None and it.context in train_contexts:
            bad(it, "context có trong train")

    # Câu E3b phải khác câu gốc đúng ở chỗ mất đáp án.
    by_id = {it.qid: it for it in items}
    for it in items:
        if it.subset == "E3b" and it.role == "perturbed":
            twin = by_id.get(f"{it.qid}-orig")
            if twin is None:
                bad(it, "thiếu câu gốc đi cặp")
                continue
            norm = f" {normalize_answer(it.context)} "
            for a in twin.answers:
                if a in it.context or (normalize_answer(a) and f" {normalize_answer(a)} " in norm):
                    bad(it, f"đáp án gốc vẫn còn trong context: {a!r}")
        if it.subset == "E2c" and it.role == "perturbed":
            fake = it.note.removeprefix("câu nhiễu: ")
            if any(a in fake for a in it.answers):
                bad(it, "câu nhiễu chứa đáp án vàng")

    per_subset: dict[str, dict] = {}
    for sub in SUBSETS:
        group = [it for it in items if it.subset == sub]
        if not group:
            continue
        main = [it for it in group if it.role != "original"]
        per_subset[sub] = {
            "description": SUBSETS[sub],
            "n_items": len(main),
            "n_original_twins": len(group) - len(main),
            "n_answerable": sum(not it.is_impossible for it in main),
            "n_impossible": sum(it.is_impossible for it in main),
            "distinct_contexts": len({it.context for it in main}),
            "distinct_source_articles": len({it.title for it in main}),
            "distinct_questions": len({it.question for it in main}),
            "max_items_per_context": max(Counter(it.context for it in main).values()),
        }

    per_category = {}
    for code in CATEGORIES:
        group = [it for it in items if it.category == code and it.role != "original"]
        per_category[code] = {
            "name": CATEGORIES[code],
            "n_items": len(group),
            "n_original_twins": sum(it.category == code and it.role == "original" for it in items),
            "n_answerable": sum(not it.is_impossible for it in group),
            "n_impossible": sum(it.is_impossible for it in group),
            # Điểm của hệ thống "luôn từ chối" trên các câu thử (không tính câu gốc).
            # Với E3 con số này là 100 theo định nghĩa — chỉ số chính của E3 vì thế là
            # độ nhất quán theo cặp (score_report), nơi "luôn từ chối" được 0.
            "always_abstain_em": round(100 * sum(it.is_impossible for it in group) / len(group), 2)
            if group else None,
        }

    return {
        "n_items": len(items),
        "n_answerable": sum(not it.is_impossible for it in items),
        "n_impossible": sum(it.is_impossible for it in items),
        "distinct_contexts": len({it.context for it in items}),
        "distinct_source_questions": len({it.source_qid for it in items}),
        "by_category": per_category,
        "by_subset": per_subset,
        "n_violations": len(violations),
        "violations": violations[:50],
    }


# ── vào/ra ───────────────────────────────────────────────────────────────────────


def to_squad(items: Sequence[StressItem], version: str = "stress-v2") -> dict:
    """Ghi ra lược đồ SQuAD-2.0; mỗi context mang đúng các câu hỏi của nó.

    Siêu dữ liệu stress-test nằm ở khoá ``stress`` của từng câu hỏi — các loader
    SQuAD bình thường bỏ qua khoá này.
    """
    by_title: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_title[it.title][it.context].append({
            "id": it.qid,
            "question": it.question,
            "is_impossible": it.is_impossible,
            "answers": {"text": list(it.answers), "answer_start": list(it.answer_starts)},
            "stress": {
                "category": it.category, "subset": it.subset, "role": it.role,
                "source_qid": it.source_qid, "pair_id": it.pair_id, "note": it.note,
            },
        })
    return {
        "version": version,
        "source": "UIT-ViQuAD 2.0 validation",
        "categories": CATEGORIES,
        "subsets": SUBSETS,
        "data": [
            {"title": t, "paragraphs": [{"context": c, "qas": qas} for c, qas in ctxs.items()]}
            for t, ctxs in by_title.items()
        ],
    }


def load_stress_v2(path: str | Path) -> tuple[list[Example], dict[str, dict]]:
    """``(examples, meta_by_qid)`` — ``meta`` là khoá ``stress`` của từng câu."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    examples: list[Example] = []
    meta: dict[str, dict] = {}
    for article in payload["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                texts = list(qa["answers"]["text"])
                starts = qa["answers"]["answer_start"]
                examples.append(Example(
                    qid=qa["id"], question=qa["question"], context=para["context"],
                    title=article["title"], answers=texts,
                    answer_start=starts[0] if texts else -1,
                    is_impossible=bool(qa["is_impossible"]),
                ))
                meta[qa["id"]] = qa["stress"]
    return examples, meta


# ── chấm điểm ────────────────────────────────────────────────────────────────────


def score_report(predictions: dict[str, str], examples: Sequence[Example],
                 meta: dict[str, dict]) -> dict:
    """Điểm theo nhóm, theo tập con và theo CẶP (perturbation ↔ câu gốc).

    * ``by_category`` — mọi câu của nhóm TRỪ câu gốc đi cặp (câu gốc là đối chứng,
      không phải phép thử).
    * ``by_subset`` — như trên, cho từng tập con; tập con có cặp thêm ``original``
      (điểm trên câu gốc) và ``pairs``:
        ``both_correct``  đúng cả câu gốc lẫn câu biến đổi (nhất quán),
        ``broken``        đúng câu gốc nhưng sai câu biến đổi — lỗi do chính phép
                          biến đổi gây ra, con số chính của phép thử,
        ``fixed``         sai câu gốc nhưng đúng câu biến đổi.
    """
    from mrc.metrics import evaluate as evaluate_metrics

    by_id = {e.qid: e for e in examples}

    def score(qids: list[str]) -> dict:
        s = evaluate_metrics({q: predictions[q] for q in qids},
                             {q: list(by_id[q].answers) for q in qids})
        return {"EM": round(s["EM"], 2), "F1": round(s["F1"], 2), "n": s["count"],
                "n_answerable": s["n_answerable"], "n_impossible": s["n_impossible"],
                "_per_item": s["per_item"]}

    def public(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    out: dict = {"by_category": {}, "by_subset": {}}
    for code in CATEGORIES:
        qids = [q for q, m in meta.items() if m["category"] == code and m["role"] != "original"]
        if qids:
            out["by_category"][code] = public(score(qids))
    for sub in SUBSETS:
        main = [q for q, m in meta.items() if m["subset"] == sub and m["role"] != "original"]
        if not main:
            continue
        s = score(main)
        entry = public(s)
        twins = {m["pair_id"]: q for q, m in meta.items()
                 if m["subset"] == sub and m["role"] == "original"}
        if twins:
            o = score(list(twins.values()))
            entry["original"] = public(o)
            both = broken = fixed = 0
            for q in main:
                t = twins.get(meta[q]["pair_id"])
                if t is None:
                    continue
                a, b = o["_per_item"][t]["em"], s["_per_item"][q]["em"]
                both += a == 1 and b == 1
                broken += a == 1 and b == 0
                fixed += a == 0 and b == 1
            n = len(main)
            entry["pairs"] = {"n": n, "both_correct": both, "broken": broken, "fixed": fixed,
                              "broken_pct_of_original_correct":
                                  round(100 * broken / (both + broken), 2) if both + broken else None}
        out["by_subset"][sub] = entry
    return out
