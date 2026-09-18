# -*- coding: utf-8 -*-
"""
Span decoding for extractive MRC.

Replaces the placeholder in train_mrc_transformer.py, which compared literal
strings of the form f"span_{start}_{end}" and therefore never measured Exact
Match or F1 at all.

Follows the reference SQuAD postprocessing: for each example, take the n_best
highest start and end logits, keep spans that are valid under the feature's
offset mapping and the max answer length, and slice the answer text out of the
original context. SQuAD 2.0 style unanswerability is supported via the null
(CLS) score.

Deliberately dependency-light: logits may be plain lists or numpy arrays, so
the decoder and its tests run without torch, transformers or numpy installed.
That matters because evaluation correctness has to be verified before any GPU
time is spent on training.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "compute_span_metrics",
    "desegment",
    "postprocess_qa_predictions",
    "select_best_span",
]


# ---------------------------------------------------------------------------
# Vietnamese word segmentation
# ---------------------------------------------------------------------------
def desegment(text: str) -> str:
    """Turn word-segmented Vietnamese back into plain text.

    Word-level models such as PhoBERT consume segmented input, where syllables
    inside one word are joined by underscores ("Hà_Nội"). Gold answers are
    stored unsegmented ("Hà Nội").

    This matters more than it looks. ``eval_metrics.normalize_text`` strips
    punctuation, and "_" is punctuation, so an unconverted prediction collapses
    to "hànội" while the gold answer normalizes to "hà nội". The two never
    match: a perfectly correct PhoBERT answer scores EM=0 and F1=0.0. Left
    unfixed, this reads as catastrophic model failure rather than an evaluation
    bug, and it inverts any monolingual-vs-multilingual comparison.

    Always apply this to predictions from a word-level model before scoring.
    It is a no-op for subword models such as XLM-R.
    """
    if not text:
        return text
    return text.replace("_", " ")


# ---------------------------------------------------------------------------
# Span selection
# ---------------------------------------------------------------------------
def _top_k_indices(logits: Sequence[float], k: int) -> List[int]:
    """Indices of the k largest values, highest first. Pure Python."""
    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:k]


def select_best_span(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offset_mapping: Sequence[Optional[Sequence[int]]],
    context: str,
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> Tuple[str, float, Optional[Tuple[int, int]]]:
    """Decode the highest-scoring valid answer span from one feature.

    A candidate (start, end) is rejected when it falls outside the context
    (offset is None), when end precedes start, or when the span is longer than
    ``max_answer_length`` tokens. These guards are what stop the decoder
    returning question text or runaway spans.

    Returns ``(answer_text, score, char_span)``; ``("", -inf, None)`` when no
    candidate survives.
    """
    best_text = ""
    best_score = float("-inf")
    best_span: Optional[Tuple[int, int]] = None

    for start_index in _top_k_indices(start_logits, n_best_size):
        for end_index in _top_k_indices(end_logits, n_best_size):
            # Out of range for this feature.
            if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                continue
            # Not part of the context (question tokens and padding are None).
            if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                continue
            # Degenerate or reversed span.
            if end_index < start_index:
                continue
            # Too long to be a plausible answer.
            if end_index - start_index + 1 > max_answer_length:
                continue

            score = float(start_logits[start_index]) + float(end_logits[end_index])
            if score > best_score:
                char_start = offset_mapping[start_index][0]
                char_end = offset_mapping[end_index][1]
                best_score = score
                best_text = context[char_start:char_end]
                best_span = (char_start, char_end)

    return best_text, best_score, best_span


# ---------------------------------------------------------------------------
# Full postprocessing
# ---------------------------------------------------------------------------
def postprocess_qa_predictions(
    examples: Sequence[Dict[str, Any]],
    features: Sequence[Dict[str, Any]],
    all_start_logits: Sequence[Sequence[float]],
    all_end_logits: Sequence[Sequence[float]],
    n_best_size: int = 20,
    max_answer_length: int = 30,
    version_2_with_negative: bool = True,
    null_score_diff_threshold: float = 0.0,
    word_level: bool = False,
) -> List[Dict[str, Any]]:
    """Convert model logits into text predictions.

    One example can produce several features when a long context is split by
    a sliding window, so scores are compared across all of an example's
    features and the best span wins.

    Args:
        examples: dicts with ``id`` and ``context``.
        features: dicts with ``example_id`` and ``offset_mapping``.
        all_start_logits / all_end_logits: per-feature logit sequences.
        version_2_with_negative: allow "no answer" using the CLS/null score.
        null_score_diff_threshold: raise to abstain more readily.
        word_level: set True for PhoBERT and other segmented-input models so
            predictions are de-segmented before scoring. See ``desegment``.

    Returns:
        ``[{"id", "prediction_text", "confidence", "no_answer_score"}]`` —
        the schema ``error_analysis.py`` consumes.
    """
    if len(features) != len(all_start_logits) or len(features) != len(all_end_logits):
        raise ValueError(
            f"logit count must match feature count: {len(features)} features, "
            f"{len(all_start_logits)} start, {len(all_end_logits)} end"
        )

    # Group feature indices by the example they came from.
    features_per_example = collections.defaultdict(list)
    for feature_index, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(feature_index)

    predictions: List[Dict[str, Any]] = []

    for example in examples:
        example_id = example["id"]
        context = example.get("context", "")
        feature_indices = features_per_example.get(example_id, [])

        best_text = ""
        best_score = float("-inf")
        min_null_score = float("inf")

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            # Index 0 is CLS: the model's score for predicting no answer.
            if len(start_logits) > 0 and len(end_logits) > 0:
                null_score = float(start_logits[0]) + float(end_logits[0])
                min_null_score = min(min_null_score, null_score)

            text, score, _span = select_best_span(
                start_logits=start_logits,
                end_logits=end_logits,
                offset_mapping=offset_mapping,
                context=context,
                n_best_size=n_best_size,
                max_answer_length=max_answer_length,
            )
            if score > best_score:
                best_score = score
                best_text = text

        # No feature produced a usable span.
        if best_score == float("-inf"):
            best_text, best_score = "", 0.0
        if min_null_score == float("inf"):
            min_null_score = 0.0

        # Abstain when the null span outscores the best real span.
        if version_2_with_negative:
            if min_null_score - best_score > null_score_diff_threshold:
                best_text = ""

        if word_level and best_text:
            best_text = desegment(best_text)

        predictions.append(
            {
                "id": example_id,
                "prediction_text": best_text,
                "confidence": round(float(best_score), 6),
                "no_answer_score": round(float(min_null_score), 6),
            }
        )

    return predictions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
from eval_metrics import compute_exact_match, compute_f1  # noqa: E402

def compute_span_metrics(predictions, references, word_level: bool = False):
    """Score decoded text spans with SQuAD Exact Match and token F1.

    Args:
        predictions: [{"id", "prediction_text"}] from postprocess_qa_predictions.
        references:  [{"id", "answers": {"text": [...]}}]. An empty text list
                     means the question is unanswerable.
        word_level:  True for segmented-input models (PhoBERT). See
                     qa_postprocessing.desegment for why this is not optional.

    A prediction is scored against its best-matching gold answer, as SQuAD does
    when several annotations exist.
    """
    ref_by_id = {r["id"]: r for r in references}
    total = em_sum = f1_sum = 0

    for pred in predictions:
        ref = ref_by_id.get(pred["id"])
        if ref is None:
            continue
        total += 1

        text = pred.get("prediction_text", "")
        if word_level:
            text = desegment(text)

        golds = ref.get("answers", {}).get("text", [])

        if not golds:
            # Unanswerable: credit only an empty prediction.
            correct = 1.0 if text.strip() == "" else 0.0
            em_sum += correct
            f1_sum += correct
            continue

        em_sum += max(compute_exact_match(text, g) for g in golds)
        f1_sum += max(compute_f1(text, g) for g in golds)

    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "total": 0}

    return {
        "exact_match": round(em_sum / total * 100, 4),
        "f1": round(f1_sum / total * 100, 4),
        "total": total,
    }
