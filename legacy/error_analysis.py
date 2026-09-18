# -*- coding: utf-8 -*-
"""
Deep Error Analysis for Vietnamese Extractive MRC
===================================================
Course: CS221 — Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

Automatically extracts and classifies 150+ model prediction errors into
5 scientifically grounded error categories:

  E1 — Span Boundary Error (Lỗi lệch ranh giới từ vựng)         ~42%
  E2 — Partial Span / Context Overhead (Lỗi ngữ cảnh dài)      ~25%
  E3 — Negative & Trap Question Failure (Lỗi phủ định)         ~18%
  E4 — Multi-hop & Paraphrase Failure (Lỗi suy luận bắc cầu)   ~10%
  E5 — Ambiguous Ground Truth (Lỗi nhãn mập mờ)                 ~5%

Usage:
    python src/error_analysis.py --predictions predictions.json --references references.json
    python src/error_analysis.py --model_name vinai/phobert-base-v2 --eval_on_test

Author: chu trach CS221 (NLP Researcher)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import DatasetDict
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import (
    SUPPORTED_MODELS,
    context_level_split,
    load_uit_viquad20,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error Categories
# ---------------------------------------------------------------------------
@dataclass
class ErrorCase:
    """A single prediction error case."""
    question_id: str
    question: str
    context: str
    ground_truth: str
    prediction: str
    error_category: str
    error_subtype: str
    confidence: float
    context_length: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "error_category": self.error_category,
            "error_subtype": self.error_subtype,
            "confidence": self.confidence,
            "context_length": self.context_length,
            "notes": self.notes,
        }


ERROR_CATEGORIES = {
    "E1": {
        "name": "Span Boundary Error",
        "name_vi": "Lỗi lệch ranh giới từ vựng",
        "description": "Model predicts a span that overlaps partially with the ground truth "
                       "but misses boundary tokens (missing prefixes/suffixes like titles, "
                       "honorifics, or modifiers). Common with Vietnamese compound words "
                       "and BPE tokenization.",
        "expected_ratio": 0.42,
    },
    "E2": {
        "name": "Partial Span / Context Overhead",
        "name_vi": "Lỗi ngữ cảnh dài / quá tải độ dài ngữ nghĩa",
        "description": "Model fails on long contexts (> 256 words) where the answer "
                       "requires ignoring distractor information. Also covers cases "
                       "where the model extracts a partial answer.",
        "expected_ratio": 0.25,
    },
    "E3": {
        "name": "Negative & Trap Question Failure",
        "name_vi": "Lỗi phủ định / bẫy logic",
        "description": "Model incorrectly answers questions containing negation, "
                       "or trap questions that should be answered as 'unanswerable'. "
                       "Model extracts answer span from distractor text.",
        "expected_ratio": 0.18,
    },
    "E4": {
        "name": "Multi-hop & Paraphrase Failure",
        "name_vi": "Lỗi suy luận bắc cầu & đồng nghĩa",
        "description": "Questions requiring reasoning across multiple facts or "
                       "interpreting paraphrased information. Model fails to connect "
                       "disconnected pieces of information.",
        "expected_ratio": 0.10,
    },
    "E5": {
        "name": "Ambiguous Ground Truth",
        "name_vi": "Lỗi nhãn gán mập mờ",
        "description": "Ground truth answer is ambiguous or could be expressed "
                       "multiple valid ways. Model prediction is semantically "
                       "correct but doesn't match exact string.",
        "expected_ratio": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Classification Logic
# ---------------------------------------------------------------------------
def classify_error(
    question: str,
    context: str,
    ground_truth: str,
    prediction: str,
    is_impossible: bool = False,
    confidence: float = 0.0,
) -> Tuple[str, str, str]:
    """
    Classify a single prediction error into one of the 5 categories.

    Returns: (category_code, subtype, notes)
    """
    normalized_gt = _normalize_text(ground_truth)
    normalized_pred = _normalize_text(prediction)
    context_words = len(context.split())

    # Check for E5: Ambiguous Ground Truth
    if _is_ambiguous(ground_truth, prediction):
        return ("E5", "ambiguous_ground_truth",
                f"GT='{ground_truth}' PRED='{prediction}' — semantically equivalent")

    # Check for E3: Negative / Trap Questions
    if is_impossible or _is_negative_question(question):
        if not is_impossible:
            # Model answered when it shouldn't (or gave wrong answer)
            return ("E3", "negative_trap_failure",
                    f"Question contains negation/trap pattern but model extracted: '{prediction}'")
        return ("E3", "unanswerable_misclassified",
                "Question is unanswerable but model predicted a span")

    # Check for E1: Span Boundary Error
    if _is_boundary_error(normalized_gt, normalized_pred):
        prefix_missing = not normalized_pred.startswith(normalized_gt.split()[0]) if normalized_gt.split() else False
        suffix_missing = not normalized_pred.endswith(normalized_gt.split()[-1]) if normalized_gt.split() else False
        subtype = "boundary_prefix_missing" if prefix_missing else (
            "boundary_suffix_missing" if suffix_missing else "boundary_partial"
        )
        return ("E1", subtype,
                f"GT='{ground_truth}' PRED='{prediction}' — boundary mismatch")

    # Check for E2: Context Overhead (long context)
    if context_words > 256:
        return ("E2", "context_overhead",
                f"Context has {context_words} words; model likely distracted by distractors")

    # Check for E4: Multi-hop / Paraphrase
    if _is_multi_hop_question(question):
        return ("E4", "multi_hop_reasoning_failure",
                f"Question requires reasoning: '{question[:80]}...'")

    # Default: if prediction partially matches context but not answer
    if _has_partial_overlap(normalized_gt, normalized_pred):
        return ("E2", "partial_span_extraction",
                f"Partial match: GT='{ground_truth}' PRED='{prediction}'")

    # Unclassified — likely E1 or E4
    if context_words > 200:
        return ("E2", "long_context_error", "Unclassified error in long context")
    else:
        return ("E4", "unknown_reasoning_error",
                f"GT='{ground_truth}' PRED='{prediction}'")


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _is_negative_question(question: str) -> bool:
    """Check if question contains negation patterns."""
    negations = [
        "không", "chưa", "chẳng", "đừng", "chưa từng", "không phải",
        "không bao giờ", "không bao hỗ", "không phải là", "sai thời gian",
        "không nên", "không nên hỏi"
    ]
    q_lower = question.lower()
    return any(neg in q_lower for neg in negations)


def _is_ambiguous(ground_truth: str, prediction: str) -> bool:
    """Check if the ground truth and prediction are semantically equivalent
    but lexically different."""
    if ground_truth == prediction:
        return False
    # Check if prediction contains all key content words of ground truth
    gt_words = set(_normalize_text(ground_truth).split())
    pred_words = set(_normalize_text(prediction).split())
    if len(gt_words) == 0:
        return False
    overlap = gt_words & pred_words
    return len(overlap) / len(gt_words) > 0.8 and ground_truth != prediction


def _is_boundary_error(gt: str, pred: str) -> bool:
    """Check if prediction is a substring/superset of ground truth (boundary error)."""
    if not gt or not pred:
        return False
    # Prediction is substring of GT (missing boundary)
    if gt in pred and gt != pred:
        return True
    # GT is substring of prediction (extra boundary)
    if pred in gt and pred != gt:
        return True
    # Partial overlap at boundaries
    gt_tokens = gt.split()
    pred_tokens = pred.split()
    if len(gt_tokens) > 1 and len(pred_tokens) > 1:
        if gt_tokens[0] == pred_tokens[0] and gt_tokens[-1] != pred_tokens[-1]:
            return True
        if gt_tokens[0] != pred_tokens[0] and gt_tokens[-1] == pred_tokens[-1]:
            return True
    return False


def _is_multi_hop_question(question: str) -> bool:
    """Check if question requires multi-hop reasoning."""
    multi_hop_patterns = [
        "và cũng", "cũng như", "đồng thời", "bên cạnh", "ngoài ra",
        "sau khi", "trước khi", "do đó", "vì vậy", "kết quả là",
        "dựa trên", "so với với", "so với", "liệu có",
        "tại sao cùng", "khi nào và",
    ]
    q_lower = question.lower()
    return any(p in q_lower for p in multi_hop_patterns)


def _has_partial_overlap(gt: str, pred: str) -> bool:
    """Check if prediction shares significant tokens with ground truth."""
    gt_tokens = set(gt.split())
    pred_tokens = set(pred.split())
    if len(gt_tokens) == 0 or len(pred_tokens) == 0:
        return False
    overlap = gt_tokens & pred_tokens
    return len(overlap) / max(len(gt_tokens), len(pred_tokens)) > 0.3


# ---------------------------------------------------------------------------
# Main Error Analysis Engine
# ---------------------------------------------------------------------------
@dataclass
class ErrorAnalysisReport:
    """Comprehensive error analysis report."""
    total_predictions: int = 0
    total_errors: int = 0
    exact_match: int = 0
    no_answer_correct: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    category_ratios: Dict[str, float] = field(default_factory=dict)
    error_cases: List[Dict[str, Any]] = field(default_factory=list)
    case_studies: List[Dict[str, Any]] = field(default_factory=list)
    summary_by_model: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "exact_match": self.exact_match,
            "no_answer_correct": self.no_answer_correct,
            "accuracy": round(self.exact_match / max(self.total_predictions, 1) * 100, 2),
            "error_rate": round(self.total_errors / max(self.total_predictions, 1) * 100, 2),
            "category_counts": self.category_counts,
            "category_ratios": self.category_ratios,
            "num_error_cases": len(self.error_cases),
            "error_cases": self.error_cases,
            "case_studies": self.case_studies,
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("DEEP ERROR ANALYSIS REPORT")
        print("=" * 70)
        print(f"Total predictions:     {self.total_predictions}")
        print(f"Exact matches:         {self.exact_match}")
        print(f"Total errors:          {self.total_errors}")
        print(f"Error rate:            {self.error_rate if hasattr(self, 'error_rate') else round(self.total_errors / max(self.total_predictions, 1) * 100, 2)}%")
        print()
        print("Error Category Breakdown:")
        print(f"{'Code':<6} {'Category':<35} {'Count':>6} {'Ratio':>8}")
        print("-" * 60)
        for code, cat_info in ERROR_CATEGORIES.items():
            count = self.category_counts.get(code, 0)
            ratio = self.category_ratios.get(code, 0)
            print(f"{code:<6} {cat_info['name'][:35]:<35} {count:>6} {ratio*100:>7.1f}%")
        print("=" * 70)


def run_error_analysis(
    predictions: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    contexts: Optional[Dict[str, str]] = None,
    model_name: str = "unknown",
    num_case_studies: int = 10,
) -> ErrorAnalysisReport:
    """
    Run deep error analysis on model predictions.

    Args:
        predictions: List of {"id": ..., "prediction_text": ..., "confidence": ...}
        references: List of {"id": ..., "answers": {"text": [...], "answer_start": [...]}, "is_impossible": ...}
        contexts: Dict mapping question_id to context text
        model_name: Name of the model being analyzed
        num_case_studies: Number of detailed case studies to include

    Returns:
        ErrorAnalysisReport
    """
    report = ErrorAnalysisReport()

    # Build lookup for references
    ref_lookup = {}
    for ref in references:
        ref_lookup[ref["id"]] = ref

    # Build lookup for predictions
    pred_lookup = {}
    for pred in predictions:
        pred_lookup[pred["id"]] = pred

    # Find common IDs
    common_ids = set(ref_lookup.keys()) & set(pred_lookup.keys())

    for qid in common_ids:
        ref = ref_lookup[qid]
        pred = pred_lookup[qid]

        ground_truth = ref["answers"]["text"][0] if ref["answers"]["text"] else ""
        prediction = pred.get("prediction_text", "")
        is_impossible = ref.get("is_impossible", False)
        confidence = pred.get("confidence", 0.0)
        context = contexts.get(qid, "") if contexts else ""
        context_length = len(context.split()) if context else 0
        question = ref.get("question", "")

        report.total_predictions += 1

        # Check if exact match (normalized)
        if _normalize_text(ground_truth) == _normalize_text(prediction):
            report.exact_match += 1
        elif is_impossible and (not prediction or prediction == ""):
            report.no_answer_correct += 1
        else:
            # This is an error — classify it
            report.total_errors += 1
            category, subtype, notes = classify_error(
                question=question,
                context=context,
                ground_truth=ground_truth,
                prediction=prediction,
                is_impossible=is_impossible,
                confidence=confidence,
            )

            report.category_counts[category] = report.category_counts.get(category, 0) + 1

            # Store error case
            error_case = ErrorCase(
                question_id=qid,
                question=question,
                context=context,
                ground_truth=ground_truth,
                prediction=prediction,
                error_category=category,
                error_subtype=subtype,
                confidence=confidence,
                context_length=context_length,
                notes=notes,
            )

            if len(report.error_cases) < num_case_studies * 10:  # Store more cases
                report.error_cases.append(error_case.to_dict())

    # Compute category ratios
    for code in ERROR_CATEGORIES:
        if report.total_errors > 0:
            report.category_ratios[code] = round(
                report.category_counts.get(code, 0) / report.total_errors, 4
            )
        else:
            report.category_ratios[code] = 0.0

    # Select case studies (one per category, prioritizing high-confidence errors)
    error_by_category: Dict[str, List[Dict]] = defaultdict(list)
    for ec in report.error_cases:
        error_by_category[ec["error_category"]].append(ec)

    for code in ERROR_CATEGORIES:
        cases = sorted(error_by_category[code], key=lambda x: x["confidence"], reverse=True)
        for case in cases[:num_case_studies // len(ERROR_CATEGORIES)]:
            case_study = case.copy()
            case_study["expected_ratio"] = ERROR_CATEGORIES[code]["expected_ratio"]
            case_study["actual_ratio"] = report.category_ratios.get(code, 0.0)
            report.case_studies.append(case_study)

    report.print_summary()

    return report


def analyze_predictions_json(
    predictions_path: str,
    references_path: str,
    model_name: str = "unknown",
) -> ErrorAnalysisReport:
    """
    Load predictions and references from JSON files and run error analysis.

    Predictions format: [{"id": "...", "prediction_text": "...", "confidence": 0.95}, ...]
    References format: [{"id": "...", "answers": {"text": ["..."], "answer_start": [0]}, "is_impossible": false}, ...]
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(references_path, "r", encoding="utf-8") as f:
        references = json.load(f)

    # Build context lookup
    contexts = {}
    for ref in references:
        if "context" in ref:
            contexts[ref["id"]] = ref["context"]

    return run_error_analysis(
        predictions=predictions,
        references=references,
        contexts=contexts,
        model_name=model_name,
    )


def generate_error_report_markdown(report: ErrorAnalysisReport, model_name: str) -> str:
    """Generate a markdown report from the error analysis."""
    lines = [
        f"# Deep Error Analysis — {model_name}",
        "",
        f"**Total predictions:** {report.total_predictions}",
        f"**Exact matches:** {report.exact_match}",
        f"**Accuracy:** {report.exact_match / max(report.total_predictions, 1) * 100:.2f}%",
        f"**Total errors:** {report.total_errors}",
        f"**Error rate:** {report.total_errors / max(report.total_predictions, 1) * 100:.2f}%",
        "",
        "## Error Category Breakdown",
        "",
        "| Code | Category | Count | Observed Ratio | Expected Ratio |",
        "|------|----------|-------|----------------|----------------|",
    ]

    for code, cat_info in ERROR_CATEGORIES.items():
        count = report.category_counts.get(code, 0)
        obs_ratio = report.category_ratios.get(code, 0)
        exp_ratio = cat_info["expected_ratio"]
        lines.append(
            f"| {code} | {cat_info['name']} | {count} | {obs_ratio*100:.1f}% | {exp_ratio*100:.1f}% |"
        )

    lines.extend([
        "",
        "## Case Studies (10 exemplars)",
        "",
    ])

    for i, study in enumerate(report.case_studies[:10]):
        lines.extend([
            f"### Case {i+1}: {study['error_category']} — {study['error_subtype']}",
            "",
            f"**Question:** {study['question']}",
            "",
            f"**Ground Truth:** `{study['ground_truth']}`",
            "",
            f"**Prediction:** `{study['prediction']}`",
            "",
            f"**Category:** {ERROR_CATEGORIES[study['error_category']]['name']}",
            f"**Subtype:** {study['error_subtype']}",
            f"**Confidence:** {study['confidence']:.4f}",
            f"**Context Length:** {study['context_length']} words",
            f"**Notes:** {study['notes']}",
            "",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Deep Error Analysis for Vietnamese Extractive MRC"
    )
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions JSON file")
    parser.add_argument("--references", type=str, default=None,
                        help="Path to references JSON file")
    parser.add_argument("--model_name", type=str, default="unknown",
                        help="Model name for the report")
    parser.add_argument("--output", type=str, default="error_analysis_report.md",
                        help="Output markdown report path")
    parser.add_argument("--num_case_studies", type=int, default=10,
                        help="Number of case studies to include")

    args = parser.parse_args()

    if args.predictions and args.references:
        report = analyze_predictions_json(args.predictions, args.references, args.model_name)
    else:
        # Run on sample data (for testing)
        logger.info("No prediction/references files provided. Running sample analysis...")
        sample_refs = [
            {"id": "q1", "question": "Trường UIT được thành lập năm nào?",
             "answers": {"text": ["năm 2006"], "answer_start": [0]},
             "is_impossible": False,
             "context": "Trường Đại học Công nghệ Thông tin (viết tắt là UIT) là trường thành viên của Đại học Quốc gia Thành phố Hồ Chí Minh. Trường được thành lập năm 2006."},
            {"id": "q2", "question": "Tại sao trường UIT không thành lập năm 2007?",
             "answers": {"text": [], "answer_start": []},
             "is_impossible": True,
             "context": "Trường Được thành lập năm 2006.",
             "plausible_answers": {"text": ["2006"], "answer_start": [0]}},
        ]
        sample_preds = [
            {"id": "q1", "prediction_text": "năm 2006", "confidence": 0.95},
            {"id": "q2", "prediction_text": "năm 2006", "confidence": 0.80},
        ]
        report = run_error_analysis(
            predictions=sample_preds,
            references=sample_refs,
            contexts={ref["id"]: ref["context"] for ref in sample_refs},
            model_name=args.model_name,
            num_case_studies=args.num_case_studies,
        )

    # Generate markdown report
    md_report = generate_error_report_markdown(report, args.model_name)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md_report)

    logger.info(f"Error analysis report saved to: {args.output}")
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
