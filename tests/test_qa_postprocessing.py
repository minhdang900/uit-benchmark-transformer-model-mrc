# -*- coding: utf-8 -*-
"""
AC-4 gate: 20 hand-checked examples must match computed EM/F1.

This suite is the checkpoint that has to pass before any GPU time is spent.
Training against the old placeholder metric — which compared literal
f"span_{start}_{end}" strings — would have burned the scarcest resource on
numbers that measure nothing.

Runs on the standard library alone: no torch, transformers, numpy or pytest.

    python3 tests/test_qa_postprocessing.py
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from eval_metrics import compute_exact_match, compute_f1  # noqa: E402
from qa_postprocessing import (  # noqa: E402
    desegment,
    postprocess_qa_predictions,
    select_best_span,
)


def peaked(length: int, index: int, value: float = 10.0) -> list:
    """Logit vector that is flat at 0.0 except for one peak."""
    logits = [0.0] * length
    logits[index] = value
    return logits


# Context shared by the span-decoding cases.
# Char offsets: "Hà Nội" = [0,6); "thủ đô" = [10,16); "Việt Nam" = [21,29)
CONTEXT = "Hà Nội là thủ đô của Việt Nam"

# offset_mapping[0] is CLS (None), then three context tokens.
OFFSETS = [None, [0, 6], [10, 16], [21, 29]]


class TestDesegment(unittest.TestCase):
    """The bug that would have silently zeroed every PhoBERT score."""

    def test_1_underscore_becomes_space(self):
        self.assertEqual(desegment("Hà_Nội"), "Hà Nội")

    def test_2_multi_syllable_compound(self):
        self.assertEqual(desegment("thành_phố Hồ_Chí_Minh"), "thành phố Hồ Chí Minh")

    def test_3_empty_string_is_safe(self):
        self.assertEqual(desegment(""), "")

    def test_4_unsegmented_text_unchanged(self):
        self.assertEqual(desegment("Hà Nội"), "Hà Nội")

    def test_5_segmented_prediction_scores_zero_without_desegment(self):
        """Regression guard: this is why desegment exists."""
        self.assertEqual(compute_exact_match("Hà_Nội", "Hà Nội"), 0)
        self.assertEqual(compute_f1("Hà_Nội", "Hà Nội"), 0.0)

    def test_6_desegmented_prediction_scores_perfect(self):
        self.assertEqual(compute_exact_match(desegment("Hà_Nội"), "Hà Nội"), 1)
        self.assertEqual(compute_f1(desegment("Hà_Nội"), "Hà Nội"), 1.0)


class TestSelectBestSpan(unittest.TestCase):
    """Span validity rules."""

    def test_7_exact_single_token_span(self):
        text, score, span = select_best_span(
            peaked(4, 1), peaked(4, 1), OFFSETS, CONTEXT
        )
        self.assertEqual(text, "Hà Nội")
        self.assertEqual(span, (0, 6))
        self.assertEqual(score, 20.0)

    def test_8_multi_token_span_spans_chars(self):
        text, _, span = select_best_span(
            peaked(4, 1), peaked(4, 3), OFFSETS, CONTEXT
        )
        self.assertEqual(text, CONTEXT[0:29])
        self.assertEqual(span, (0, 29))

    def test_9_reversed_span_rejected(self):
        """end < start must never be returned as-is."""
        text, _, span = select_best_span(
            peaked(4, 3), peaked(4, 1), OFFSETS, CONTEXT
        )
        # The peak pair (start=3, end=1) is invalid, so the decoder must fall
        # back to some other valid ordering rather than returning a reversed span.
        self.assertIsNotNone(span)
        self.assertLessEqual(span[0], span[1])

    def test_10_cls_token_never_selected_as_answer(self):
        """CLS peaking must not make CLS the returned span.

        select_best_span decodes the best *context* span and nothing else.
        Choosing to abstain is the null-score comparison's job, covered by
        test_15. So with CLS peaked the decoder still returns a real context
        span — it just must never return the CLS position itself.
        """
        text, _, span = select_best_span(
            peaked(4, 0), peaked(4, 0), OFFSETS, CONTEXT
        )
        self.assertIsNotNone(span)
        # CLS carries offset None, so any returned span comes from the context.
        self.assertIn(list(span), [o for o in OFFSETS if o is not None])
        self.assertNotEqual(text, "")

    def test_11_max_answer_length_enforced(self):
        offsets = [None] + [[i, i + 1] for i in range(10)]
        text, _, _ = select_best_span(
            peaked(11, 1), peaked(11, 10), offsets, "abcdefghij", max_answer_length=3
        )
        self.assertLessEqual(len(text), 3)

    def test_12_no_valid_candidate_returns_empty(self):
        text, score, span = select_best_span([0.0], [0.0], [None], CONTEXT)
        self.assertEqual(text, "")
        self.assertEqual(score, float("-inf"))
        self.assertIsNone(span)


class TestPostprocessing(unittest.TestCase):
    """End-to-end decoding."""

    EXAMPLES = [{"id": "q1", "context": CONTEXT}]
    FEATURES = [{"example_id": "q1", "offset_mapping": OFFSETS}]

    def test_13_answerable_prediction_matches_gold(self):
        preds = postprocess_qa_predictions(
            self.EXAMPLES, self.FEATURES, [peaked(4, 1)], [peaked(4, 1)]
        )
        self.assertEqual(preds[0]["prediction_text"], "Hà Nội")
        self.assertEqual(compute_exact_match(preds[0]["prediction_text"], "Hà Nội"), 1)

    def test_14_output_schema_matches_error_analysis(self):
        preds = postprocess_qa_predictions(
            self.EXAMPLES, self.FEATURES, [peaked(4, 1)], [peaked(4, 1)]
        )
        for key in ("id", "prediction_text", "confidence"):
            self.assertIn(key, preds[0])

    def test_15_abstains_when_null_score_wins(self):
        """SQuAD 2.0: high CLS score means no answer."""
        preds = postprocess_qa_predictions(
            self.EXAMPLES, self.FEATURES,
            [peaked(4, 0, 50.0)], [peaked(4, 0, 50.0)],
            version_2_with_negative=True,
        )
        self.assertEqual(preds[0]["prediction_text"], "")

    def test_16_word_level_flag_desegments(self):
        preds = postprocess_qa_predictions(
            [{"id": "q1", "context": "Hà_Nội là thủ_đô"}],
            [{"example_id": "q1", "offset_mapping": [None, [0, 6]]}],
            [peaked(2, 1)], [peaked(2, 1)],
            word_level=True,
        )
        self.assertEqual(preds[0]["prediction_text"], "Hà Nội")

    def test_17_partial_overlap_gives_partial_f1(self):
        f1 = compute_f1("thủ đô của Việt Nam", "Việt Nam")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_18_example_with_no_features_yields_empty(self):
        preds = postprocess_qa_predictions(
            [{"id": "orphan", "context": CONTEXT}], [], [], []
        )
        self.assertEqual(preds[0]["prediction_text"], "")
        self.assertEqual(len(preds), 1)

    def test_19_mismatched_logit_count_raises(self):
        with self.assertRaises(ValueError):
            postprocess_qa_predictions(
                self.EXAMPLES, self.FEATURES, [peaked(4, 1)], []
            )

    def test_20_every_example_gets_exactly_one_prediction(self):
        examples = [{"id": f"q{i}", "context": CONTEXT} for i in range(5)]
        features = [{"example_id": f"q{i}", "offset_mapping": OFFSETS} for i in range(5)]
        preds = postprocess_qa_predictions(
            examples, features, [peaked(4, 1)] * 5, [peaked(4, 1)] * 5
        )
        self.assertEqual(len(preds), 5)
        self.assertEqual([p["id"] for p in preds], [f"q{i}" for i in range(5)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
