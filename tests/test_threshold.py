"""Ngưỡng "không có đáp án" τ — dựng lại dự đoán từ điểm từng cửa sổ, chọn τ trên dev.

Quy ước giống hệt ``TransformerQA.predict_detailed``: một cửa sổ được TRẢ LỜI khi
``best_score > null_score + τ`` tức ``delta + τ < 0`` với ``delta = null − best``;
đáp án là span điểm cao nhất trong các cửa sổ được trả lời, rỗng nếu không có.
"""
import pytest

from mrc.threshold import best_threshold, decide


def W(delta, score, answer):
    return {"delta": delta, "score": score, "answer": answer}


def test_answers_when_best_beats_null():
    assert decide([W(-1.0, 5.0, "Paris")], tau=0.0) == "Paris"


def test_abstains_when_null_wins():
    assert decide([W(2.0, 5.0, "Paris")], tau=0.0) == ""


def test_tie_at_boundary_abstains_like_predictor():
    # predictor: `best <= null + τ` -> từ chối
    assert decide([W(0.0, 5.0, "Paris")], tau=0.0) == ""


def test_positive_tau_makes_model_abstain_more():
    assert decide([W(-1.0, 5.0, "Paris")], tau=1.5) == ""


def test_negative_tau_makes_model_answer_more():
    assert decide([W(1.0, 5.0, "Paris")], tau=-1.5) == "Paris"


def test_picks_highest_score_among_answering_windows():
    ws = [W(-1.0, 3.0, "a"), W(-0.5, 7.0, "b"), W(2.0, 9.0, "c")]
    assert decide(ws, tau=0.0) == "b"


def test_changing_tau_can_change_which_window_answers():
    ws = [W(-3.0, 3.0, "a"), W(-0.5, 7.0, "b")]
    assert decide(ws, tau=1.0) == "a"


def test_window_without_null_score_always_answers():
    assert decide([W(None, 4.0, "x")], tau=100.0) == "x"


def test_no_windows_abstains():
    assert decide([], tau=0.0) == ""


def test_best_threshold_raises_tau_when_model_answers_impossible_questions():
    records = {
        "q1": [W(-0.5, 5.0, "sai")],   # impossible, model trả lời với biên độ nhỏ
        "q2": [W(-3.0, 5.0, "Paris")],  # answerable, tự tin
    }
    refs = {"q1": [], "q2": ["Paris"]}
    r = best_threshold(records, refs)
    assert 0.5 <= r["tau"] < 3.0
    assert r["f1"] == pytest.approx(100.0)


def test_best_threshold_prefers_tau_closest_to_zero_on_ties():
    records = {"q1": [W(-3.0, 5.0, "Paris")]}
    r = best_threshold(records, {"q1": ["Paris"]})
    assert r["tau"] == pytest.approx(0.0)
