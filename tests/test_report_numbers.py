"""Tầng định dạng số cho báo cáo và slide (scripts/make_report_numbers.py).

Nguyên tắc của đồ án: giá trị nào chưa có phải hiện ra (``\\missing{}`` -> ``??``
đỏ trong PDF), không bao giờ là một con số đoán — và không bao giờ làm dừng cả
lần sinh số liệu.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_report_numbers import ci, get, pval, vi  # noqa: E402

MISSING = r"\missing{}"


# ── khoảng tin cậy ───────────────────────────────────────────────────────────

def test_ci_reports_missing_when_no_interval_is_available():
    assert ci({}) == MISSING


def test_ci_reports_missing_when_article_interval_is_null_and_no_fallback():
    assert ci({"ci95_article": None}) == MISSING


def test_ci_prefers_the_article_level_interval():
    assert ci({"ci95": [1.0, 2.0], "ci95_article": [3.0, 4.0]}) == "[3{,}00; 4{,}00]"


def test_ci_falls_back_to_the_question_level_interval():
    assert ci({"ci95": [1.0, 2.0]}) == "[1{,}00; 2{,}00]"


# ── định dạng số kiểu Việt Nam ───────────────────────────────────────────────

def test_vi_uses_a_dot_as_thousands_separator_for_counts():
    assert vi(3814, 0) == "3.814"


def test_vi_uses_a_comma_as_decimal_separator():
    assert vi(30.44) == "30{,}44"


def test_vi_separates_thousands_and_decimals_in_the_same_number():
    assert vi(1234.5, 1) == "1.234{,}5"


def test_vi_reports_missing_rather_than_guessing_a_number():
    assert vi(None) == MISSING


# ── giá trị p ────────────────────────────────────────────────────────────────

def test_pval_collapses_very_small_values_to_a_threshold():
    assert pval(0.0005) == r"$<$\,0{,}001"


def test_pval_formats_ordinary_values_to_three_decimals():
    assert pval(0.9023) == "0{,}902"


def test_pval_reports_missing_rather_than_guessing():
    assert pval(None) == MISSING


# ── tra cứu lồng nhau ────────────────────────────────────────────────────────

def test_get_returns_none_when_a_key_is_absent():
    assert get({"a": {"b": 1}}, "a", "zzz") is None


def test_get_returns_none_instead_of_raising_when_a_level_is_not_a_dict():
    assert get({"a": 1}, "a", "b") is None
