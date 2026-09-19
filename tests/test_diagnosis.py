"""Bất biến của phân tích chẩn đoán, stress-test loader và baseline từ chối."""
from __future__ import annotations

import json

import pytest

from mrc.baselines import AlwaysAbstain
from mrc.diagnosis import (
    abstention_stats, classify_error, error_taxonomy, mcnemar_exact,
    paired_comparison, slice_scores,
)
from mrc.stress_test import audit_item, load_stress_test


@pytest.mark.parametrize("pred,golds,expected", [
    ("Hà Nội", ["Hà Nội"], None),
    ("", [], None),
    ("Hà Nội", [], "false_answer"),
    ("", ["Hà Nội"], "false_abstain"),
    ("thành phố Hà Nội", ["Hà Nội"], "boundary_superset"),
    ("Nội", ["Hà Nội"], "boundary_subset"),
    ("Nội là thủ đô", ["Hà Nội là"], "boundary_overlap"),
    ("Paris", ["Hà Nội"], "wrong_span"),
    ("Hà Nội.", ["Hà Nội"], None),  # chuẩn hoá bỏ dấu câu
])
def test_classify_error(pred, golds, expected):
    assert classify_error(pred, golds) == expected


def test_multi_gold_uses_best_matching_answer():
    assert classify_error("năm 1945", ["1945", "Paris"]) == "boundary_superset"


def test_taxonomy_counts_sum_to_errors():
    refs = {"a": ["x"], "b": [], "c": ["y z"], "d": ["w"]}
    preds = {"a": "x", "b": "q", "c": "y", "d": ""}
    t = error_taxonomy(preds, refs)
    assert t["n_errors"] == 3 == sum(t["counts"].values())


def test_abstention_stats_on_always_abstain():
    refs = {"a": ["x"], "b": [], "c": []}
    s = abstention_stats({q: "" for q in refs}, refs)
    assert s["abstain_rate"] == 100.0 and s["recall"] == 100.0
    assert s["precision"] == pytest.approx(66.67)


def test_slice_scores_credits_empty_on_impossible_only():
    refs = {"a": ["x"], "b": []}
    assert slice_scores({"a": "", "b": ""}, refs, ["a", "b"])["EM"] == 50.0
    assert slice_scores({}, refs, [])["count"] == 0


def test_mcnemar_symmetric_and_bounded():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 10) == 1.0
    assert mcnemar_exact(3, 17) == mcnemar_exact(17, 3) < 0.01


def test_paired_comparison_table_and_ci():
    a = {str(i): 1.0 for i in range(100)}
    b = {str(i): float(i < 80) for i in range(100)}
    r = paired_comparison(a, b)
    assert (r["both_correct"], r["only_a"], r["only_b"], r["neither"]) == (80, 20, 0, 0)
    assert r["em_diff"] == 20.0 and r["ci95"][0] <= 20.0 <= r["ci95"][1]
    assert r["mcnemar_p"] < 1e-5


def test_always_abstain_returns_empty():
    assert AlwaysAbstain().predict("ctx", "q") == ""


def test_audit_flags_answer_missing_from_context():
    qa = {"question": "Ai?", "answers": {"text": ["Bác Hồ"], "answer_start": [0]}}
    a = audit_item("Hà Nội là thủ đô.", qa)
    assert a["answerable"] and not a["answer_in_context"] and not a["valid"]


def test_audit_accepts_unanswerable_and_bad_offset():
    ctx = "Hà Nội là thủ đô."
    assert audit_item(ctx, {"question": "?", "answers": {"text": [], "answer_start": []},
                            "is_impossible": True})["valid"]
    a = audit_item(ctx, {"question": "?", "answers": {"text": ["thủ đô"], "answer_start": [0]}})
    assert a["valid"] and not a["offset_correct"]


def test_load_stress_test_skips_flat_records(tmp_path):
    squad = {"title": "t", "paragraphs": [{"context": "A là B.", "qas": [
        {"id": "q1", "question": "A là gì?", "is_impossible": False,
         "answers": {"text": ["B"], "answer_start": [99]}}]}]}
    flat = {"id": "f1", "context": "c", "question": "q", "expected_answer": "z"}
    from mrc.stress_test import CATEGORIES
    for name in CATEGORIES.values():
        (tmp_path / f"{name}.json").write_text(json.dumps({"data": [squad, flat]}))
    ex, meta, stats = load_stress_test(tmp_path)
    assert len(ex) == 5 and ex[0].answer_start == 5  # offset sai được sửa bằng find
    assert stats["E1"]["flat_records"] == 1


def test_article_bootstrap_is_wider_when_gains_cluster_in_few_articles():
    # 20 bài × 10 câu; A hơn B ở MỌI câu của 4 bài, hoà ở phần còn lại.
    # Theo câu thì trông chắc chắn; theo bài thì chỉ 4/20 bài đóng góp.
    a = {f"{g}-{i}": 1.0 for g in range(20) for i in range(10)}
    b = {f"{g}-{i}": float(g >= 4) for g in range(20) for i in range(10)}
    groups = {q: q.split("-")[0] for q in a}
    r = paired_comparison(a, b, groups=groups)
    w_q = r["ci95"][1] - r["ci95"][0]
    w_g = r["ci95_article"][1] - r["ci95_article"][0]
    assert w_g > 1.5 * w_q
    assert r["n_articles"] == 20


def test_article_bootstrap_absent_without_groups():
    r = paired_comparison({"1": 1.0}, {"1": 0.0})
    assert "ci95_article" not in r
