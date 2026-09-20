"""Bộ stress-test v2: mỗi luật dựng dữ liệu được pin bằng một ví dụ nhỏ."""
import random

import pytest

from mrc.data import Example
from mrc.stress_v2 import (
    StressItem,
    _shift_all_numbers,
    audit,
    build,
    lexical_gap,
    load_stress_v2,
    make_distractor,
    remove_answer_sentence,
    sentence_spans,
    strip_diacritics,
    to_squad,
)

CTX = ("Paris là thủ đô của Pháp. Năm 1889, tháp Eiffel được khánh thành tại Paris. "
       "Thành phố có nhiều bảo tàng. Sông Seine chảy qua trung tâm.")


def _ex(qid="q1", question="Tháp Eiffel được khánh thành tại Paris năm nào?", answer="1889",
        context=CTX, title="Paris", impossible=False):
    if impossible:
        return Example(qid=qid, question=question, context=context, title=title,
                       answers=[], answer_start=-1, is_impossible=True)
    return Example(qid=qid, question=question, context=context, title=title,
                   answers=[answer], answer_start=context.find(answer))


def _words(context):
    """Tách từ giả: mỗi âm tiết là một từ — đủ cho các test không đo E1."""
    out, pos = [], 0
    for tok in context.split():
        s = context.find(tok, pos)
        out.append((tok, s, s + len(tok)))
        pos = s + len(tok)
    return out


# ── tiện ích ────────────────────────────────────────────────────────────────────

def test_sentence_spans_cover_text_without_losing_characters():
    spans = sentence_spans(CTX)
    assert len(spans) == 4
    assert CTX[spans[1][0]:spans[1][1]] == "Năm 1889, tháp Eiffel được khánh thành tại Paris."


def test_sentence_spans_do_not_split_inside_abbreviations_or_decimals():
    text = "TP.HCM có 9,3 triệu dân. Diện tích là 2.095 km²."
    assert len(sentence_spans(text)) == 2


def test_strip_diacritics_removes_tone_and_hat_and_maps_d():
    assert strip_diacritics("Đường Hồ Chí Minh ở đâu?") == "Duong Ho Chi Minh o dau?"


def test_shifted_dates_stay_valid():
    rng = random.Random(0)
    for _ in range(50):
        out = _shift_all_numbers("ngày 31 tháng 12 năm 2020, thế kỷ 19", rng)
        day, month, year, century = (int(x) for x in __import__("re").findall(r"\d+", out))
        assert 1 <= day <= 28 and 1 <= month <= 12 and year <= 2024 and 1 <= century <= 21
        assert (day, month, year, century) != (31, 12, 2020, 19)


# ── perturbation ────────────────────────────────────────────────────────────────

def test_remove_answer_sentence_makes_question_unanswerable():
    new, removed = remove_answer_sentence(_ex())
    assert "1889" not in new
    assert removed.startswith("Năm 1889")
    assert new.startswith("Paris là thủ đô của Pháp. Thành phố")


def test_remove_answer_sentence_rejects_when_answer_repeats_elsewhere():
    ctx = CTX + " Năm 1889 cũng là năm hội chợ thế giới."
    assert remove_answer_sentence(_ex(context=ctx)) is None


def test_remove_answer_sentence_rejects_answer_spanning_two_sentences():
    ex = _ex(answer="Pháp. Năm 1889")
    assert remove_answer_sentence(ex) is None


def test_distractor_changes_subject_and_numbers_but_not_gold():
    new_ctx, fake, start = make_distractor(_ex(), random.Random(3))
    assert "Paris" not in fake and "1889" not in fake
    assert new_ctx[start:start + 4] == "1889"
    assert fake in new_ctx


def test_distractor_requires_subject_in_question():
    ex = _ex(question="Tháp Eiffel được khánh thành năm nào?")
    assert make_distractor(ex, random.Random(0)) is None


def test_distractor_skips_titles_of_unknown_entity_type():
    ex = _ex(title="Kiến", question="Kiến ... năm nào?")
    assert make_distractor(ex, random.Random(0)) is None


def test_lexical_gap():
    ctx = "Sông Seine chảy qua Paris và tháp Eiffel. Công trình này được khánh thành năm 1889."
    close = _ex(question="Công trình này được khánh thành năm nào?", context=ctx)
    far = _ex(question="Tháp Eiffel bên sông Seine ở Paris có từ năm nào?", context=ctx)
    assert not lexical_gap(close)
    assert lexical_gap(far)


# ── dựng + kiểm định ────────────────────────────────────────────────────────────

@pytest.fixture
def pool():
    exs = []
    for i in range(6):
        ctx = CTX.replace("thủ đô", f"thủ đô số {i}")
        exs.append(_ex(qid=f"a{i}", context=ctx))
        exs.append(_ex(qid=f"i{i}", question="Dân số Paris là bao nhiêu?", context=ctx,
                       impossible=True))
    return exs


def test_build_produces_gradeable_items_that_pass_audit(pool):
    items = build(pool, words_by_context={e.context: _words(e.context) for e in pool})
    report = audit(items)
    assert report["n_violations"] == 0, report["violations"]
    assert {it.subset for it in items} >= {"E2c", "E3a", "E3b", "E5"}
    for it in items:
        if it.answers:
            assert it.context[it.answer_starts[0]:].startswith(it.answers[0])


def test_every_perturbation_has_its_original_twin(pool):
    items = build(pool, words_by_context={e.context: _words(e.context) for e in pool})
    ids = {it.qid for it in items}
    for it in items:
        if it.role == "perturbed":
            assert f"{it.qid}-orig" in ids


def test_build_is_deterministic(pool):
    words = {e.context: _words(e.context) for e in pool}
    a = [it.qid + it.context for it in build(pool, words_by_context=words)]
    b = [it.qid + it.context for it in build(pool, words_by_context=words)]
    assert a == b


def test_at_most_two_items_per_context_per_subset(pool):
    items = build(pool, words_by_context={e.context: _words(e.context) for e in pool})
    report = audit(items)
    assert all(s["max_items_per_context"] <= 2 for s in report["by_subset"].values())


def test_audit_flags_answer_left_in_context():
    orig = StressItem("x-orig", "Q?", CTX, "Paris", ["1889"], [CTX.find("1889")], "E3", "E3b",
                      "original", "q", "x")
    bad = StressItem("x", "Q?", CTX, "Paris", [], [], "E3", "E3b", "perturbed", "q", "x",
                     note="câu đã xoá: ...")
    report = audit([bad, orig])
    assert report["n_violations"] == 1


def test_audit_flags_wrong_offset():
    it = StressItem("y", "Q?", CTX, "Paris", ["1889"], [0], "E4", "E4", "slice", "q")
    assert audit([it])["n_violations"] == 1


def test_audit_flags_train_overlap():
    it = StressItem("z", "Q?", CTX, "Paris", [], [], "E3", "E3a", "slice", "q")
    assert audit([it], train_contexts={CTX})["n_violations"] == 1


def test_squad_roundtrip(tmp_path, pool):
    items = build(pool, words_by_context={e.context: _words(e.context) for e in pool})
    path = tmp_path / "s.json"
    import json

    path.write_text(json.dumps(to_squad(items), ensure_ascii=False), encoding="utf-8")
    examples, meta = load_stress_v2(path)
    assert len(examples) == len(items)
    assert {e.qid for e in examples} == set(meta)
    assert all(e.is_gradeable for e in examples)


def test_score_report_pairs(pool):
    from mrc.stress_v2 import score_report

    items = build(pool, words_by_context={e.context: _words(e.context) for e in pool})
    examples = [Example(qid=it.qid, question=it.question, context=it.context, title=it.title,
                        answers=it.answers, answer_start=it.answer_starts[0] if it.answers else -1,
                        is_impossible=it.is_impossible) for it in items]
    meta = {it.qid: {"category": it.category, "subset": it.subset, "role": it.role,
                     "pair_id": it.pair_id} for it in items}
    # "Luôn từ chối": đúng mọi câu E3b bị xoá, sai mọi câu gốc -> 0 cặp nhất quán.
    abstain = score_report({e.qid: "" for e in examples}, examples, meta)
    assert abstain["by_subset"]["E3b"]["EM"] == 100.0
    assert abstain["by_subset"]["E3b"]["pairs"]["both_correct"] == 0
    # Oracle: đúng tất cả -> không cặp nào bị "broken".
    oracle = score_report({e.qid: (e.answers[0] if e.answers else "") for e in examples}, examples, meta)
    for sub in oracle["by_subset"].values():
        if "pairs" in sub:
            assert sub["pairs"]["broken"] == 0
            assert sub["pairs"]["both_correct"] == sub["pairs"]["n"]
