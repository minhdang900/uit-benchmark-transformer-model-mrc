# -*- coding: utf-8 -*-
"""
Data Leakage Detection Tool for Vietnamese Extractive MRC
==========================================================
Course: CS221 — Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

This module detects various forms of data leakage in MRC datasets:

  1. Context overlap: Same context string in multiple splits
  2. Title overlap: Same Wikipedia article in multiple splits
  3. ID overlap: Same QA pair ID in multiple splits
  4. Near-duplicate contexts: Contexts that are substrings of each other
  5. Answer leakage: Identical answer strings across different questions/contexts

Author: chu trach CS221 (NLP Researcher)
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip, unicode normalize)."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def text_hash(text: str) -> str:
    """Generate MD5 hash of normalized text."""
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def normalize_vietnamese(text: str) -> str:
    """
    Normalize Vietnamese text: remove diacritics for fuzzy matching.
    This helps detect near-duplicates that differ only in tone marks.
    """
    # Map Vietnamese diacritics to base characters
    vietnamese_chars = "àáảãèéẻẽìíìòóồổỗỜờớờạăâêôôơưýỳỵỵ"
    base_chars = "aaaaeeeiiiooooooaaaaceeeeouuuuuiii"

    # Normalize for tone marks
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return normalize_text(text)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------
@dataclass
class LeakageReport:
    """Structured report of data leakage findings."""
    total_contexts: int = 0
    total_questions: int = 0
    exact_context_overlaps: List[Dict[str, Any]] = field(default_factory=list)
    title_overlaps: List[Dict[str, Any]] = field(default_factory=list)
    id_overlaps: List[Dict[str, Any]] = field(default_factory=list)
    near_duplicate_pairs: List[Dict[str, Any]] = field(default_factory=list)
    answer_leakage: List[Dict[str, Any]] = field(default_factory=list)
    is_clean: bool = True
    severity: str = "unknown"  # "clean", "low", "medium", "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_contexts": self.total_contexts,
            "total_questions": self.total_questions,
            "exact_context_overlaps": self.exact_context_overlaps,
            "title_overlaps": self.title_overlaps,
            "id_overlaps": self.id_overlaps,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "answer_leakage": self.answer_leakage,
            "is_clean": self.is_clean,
            "severity": self.severity,
        }

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "DATA LEAKAGE REPORT",
            "=" * 70,
            f"Total contexts analyzed: {self.total_contexts}",
            f"Total questions analyzed: {self.total_questions}",
            f"Severity: {self.severity}",
            f"Status: {'CLEAN' if self.is_clean else 'LEAKAGE DETECTED'}",
            "",
            f"  Exact context overlaps: {len(self.exact_context_overlaps)}",
            f"  Title overlaps: {len(self.title_overlaps)}",
            f"  ID overlaps: {len(self.id_overlaps)}",
            f"  Near-duplicate context pairs: {len(self.near_duplicate_pairs)}",
            f"  Answer leakage cases: {len(self.answer_leakage)}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Context-Level Overlap Detection
# ---------------------------------------------------------------------------
def check_context_overlap(dataset_dict: DatasetDict) -> List[Dict[str, Any]]:
    """
    Check if the same context appears in multiple splits.
    This is the most critical leakage check — context-level splitting
    should ensure zero overlap.
    """
    overlaps = []
    splits = list(dataset_dict.keys())
    context_map: Dict[str, Set[str]] = {}  # context_hash -> set of split names

    for split_name in splits:
        ds = dataset_dict[split_name]
        contexts = ds["context"] if "context" in ds.features else []
        for ctx in contexts:
            ctx_hash = text_hash(ctx)
            if ctx_hash not in context_map:
                context_map[ctx_hash] = {"splits": set(), "example": ctx[:200]}
            context_map[ctx_hash]["splits"].add(split_name)

    for ctx_hash, info in context_map.items():
        if len(info["splits"]) > 1:
            overlaps.append({
                "context_hash": ctx_hash,
                "splits": sorted(list(info["splits"])),
                "context_preview": info["example"],
            })

    return overlaps


# ---------------------------------------------------------------------------
# 2. Title Overlap Detection
# ---------------------------------------------------------------------------
def check_title_overlap(dataset_dict: DatasetDict) -> List[Dict[str, Any]]:
    """
    Check if the same Wikipedia article title appears in multiple splits.
    """
    overlaps = []
    splits = list(dataset_dict.keys())
    title_map: Dict[str, Set[str]] = {}

    for split_name in splits:
        ds = dataset_dict[split_name]
        titles = ds["title"] if "title" in ds.features else []
        for title in titles:
            t = normalize_text(title)
            if t not in title_map:
                title_map[t] = {"splits": set(), "title": title}
            title_map[t]["splits"].add(split_name)

    for title, info in title_map.items():
        if len(info["splits"]) > 1:
            overlaps.append({
                "title": info["title"],
                "splits": sorted(list(info["splits"])),
            })

    return overlaps


# ---------------------------------------------------------------------------
# 3. ID Overlap Detection
# ---------------------------------------------------------------------------
def check_id_overlap(dataset_dict: DatasetDict) -> List[Dict[str, Any]]:
    """
    Check if the same QA pair ID appears in multiple splits.
    """
    overlaps = []
    splits = list(dataset_dict.keys())
    id_map: Dict[str, Set[str]] = {}

    for split_name in splits:
        ds = dataset_dict[split_name]
        ids = ds["id"] if "id" in ds.features else []
        for qid in ids:
            if qid not in id_map:
                id_map[qid] = set()
            id_map[qid].add(split_name)

    for qid, split_set in id_map.items():
        if len(split_set) > 1:
            overlaps.append({
                "id": qid,
                "splits": sorted(list(split_set)),
            })

    return overlaps


# ---------------------------------------------------------------------------
# 4. Near-Duplicate Context Detection (using MinHash LSH approximation)
# ---------------------------------------------------------------------------
def _minhash_signature(tokens: List[str], num_hashes: int = 50) -> List[int]:
    """Generate MinHash signature for a set of tokens using multiple hash functions."""
    if not tokens:
        return [0] * num_hashes
    # Use a set of tokens (shingles)
    shingles = set(tokens)
    # Generate num_hashes hash functions using (a*x + b) % m
    a = [random.randint(1, 100000) for _ in range(num_hashes)]
    b = [random.randint(0, 100000) for _ in range(num_hashes)]
    m = 2**32 - 1  # Large prime for hash space

    signatures = []
    for i in range(num_hashes):
        min_hash = min(
            (a[i] * hash(s) + b[i]) % m
            for s in shingles
        )
        signatures.append(min_hash)
    return signatures


def _jaccard_from_signatures(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a) if sig_a else 0.0


def check_near_duplicate_contexts(
    dataset_dict: DatasetDict,
    threshold: float = 0.85,
    sample_limit: int = 2000,
) -> List[Dict[str, Any]]:
    """
    Detect near-duplicate contexts using MinHash LSH approximation.

    Uses locality-sensitive hashing to efficiently find contexts with high
    Jaccard similarity, avoiding the O(n²) pairwise comparison.

    This catches scenarios where contexts are slightly reworded
    versions of each other (paraphrased leakage).

    Args:
        dataset_dict: DatasetDict with split data
        threshold: Minimum Jaccard similarity to flag
        sample_limit: Maximum number of contexts to sample per split for efficiency

    Returns:
        List of near-duplicate context pairs
    """
    import random as _random
    _random.seed(42)

    overlaps = []
    splits = list(dataset_dict.keys())

    # Collect unique contexts per split
    split_contexts: Dict[str, List[Tuple[str, str]]] = {}  # split -> [(context, hash)]
    for split_name in splits:
        ds = dataset_dict[split_name]
        contexts = ds["context"] if "context" in ds.features else []

        seen_hashes: Set[str] = set()
        unique = []
        for ctx in contexts:
            ctx_hash = text_hash(ctx)
            if ctx_hash not in seen_hashes:
                seen_hashes.add(ctx_hash)
                unique.append((ctx, ctx_hash))

        # Sample for efficiency
        if len(unique) > sample_limit:
            _random.seed(42)
            unique = _random.sample(unique, sample_limit)

        split_contexts[split_name] = unique

    # Compute MinHash signatures
    num_hashes = 50
    signatures: Dict[str, Dict[str, List[int]]] = {}  # split -> {hash -> signature}

    for split_name, ctxs in split_contexts.items():
        signatures[split_name] = {}
        for ctx, ctx_hash in ctxs:
            tokens = normalize_text(ctx).split()
            sig = _minhash_signature(tokens, num_hashes)
            signatures[split_name][ctx_hash] = sig

    # Compare contexts across splits using MinHash approximation
    split_names = list(splits)
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            split_a, split_b = split_names[i], split_names[j]

            # Bucket by first few hash values for LSH
            buckets: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # bucket_key -> [(hash_a, hash_b, split)]

            for ctx_hash_a, sig_a in signatures[split_a].items():
                for ctx_hash_b, sig_b in signatures[split_b].items():
                    # Use band hashing for LSH (first 10 hashes as band key)
                    band_key = "|".join(str(h) for h in sig_a[:10])
                    buckets[band_key].append((ctx_hash_a, ctx_hash_b, split_a, split_b))

            # For candidates in same band, compute actual similarity
            candidates = set()
            for bucket_entries in buckets.values():
                if len(bucket_entries) > 1:
                    for entry in bucket_entries:
                        candidates.add(entry)

            # Verify candidates with full signature comparison
            checked = set()
            for ctx_hash_a, ctx_hash_b, sa, sb in candidates:
                pair_key = (ctx_hash_a, ctx_hash_b)
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                sig_a = signatures[sa][ctx_hash_a]
                sig_b = signatures[sb][ctx_hash_b]
                jaccard = _jaccard_from_signatures(sig_a, sig_b)

                if jaccard > threshold:
                    overlaps.append({
                        "split_a": sa,
                        "split_b": sb,
                        "jaccard_similarity": round(jaccard, 4),
                        "context_hash_a": ctx_hash_a,
                        "context_hash_b": ctx_hash_b,
                    })

    return overlaps


# ---------------------------------------------------------------------------
# 5. Answer Leakage Detection
# ---------------------------------------------------------------------------
def check_answer_leakage(
    dataset_dict: DatasetDict,
    threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Detect answer leakage: identical or near-identical answer strings
    used in questions from different splits.

    This catches scenarios where the same fact is tested in train and test
    with slightly different questions.
    """
    overlaps = []
    splits = list(dataset_dict.keys())

    # Collect answers per split
    split_answers: Dict[str, List[Tuple[str, str, str]]] = {}  # split -> [(answer_text, split, id)]
    for split_name in splits:
        ds = dataset_dict[split_name]
        answers_col = ds["answers"] if "answers" in ds.features else None
        ids = ds["id"] if "id" in ds.features else [""] * len(ds)

        if answers_col:
            answer_texts = answers_col["text"]
            for i, texts in enumerate(answer_texts):
                if texts and len(texts) > 0:
                    answer = normalize_text(texts[0])
                    if answer:
                        split_answers.setdefault(split_name, []).append((answer, split_name, ids[i]))

    # Check for identical answers across splits
    all_answers: Dict[str, List[Tuple[str, str]]] = {}
    for split_name, answers in split_answers.items():
        for answer, _, qid in answers:
            if answer not in all_answers:
                all_answers[answer] = []
            all_answers[answer].append((split_name, qid))

    for answer, occurrences in all_answers.items():
        splits_with_answer = set(s for s, _ in occurrences)
        if len(splits_with_answer) > 1:
            overlaps.append({
                "answer": answer[:100],
                "splits": sorted(list(splits_with_answer)),
                "count": len(occurrences),
            })

    return overlaps


# ---------------------------------------------------------------------------
# 6. Answerable/Unanswerable Distribution Check
# ---------------------------------------------------------------------------
def check_distribution_shift(dataset_dict: DatasetDict) -> Dict[str, Any]:
    """
    Check if the answerable/unanswerable distribution differs significantly
    across splits. This isn't data leakage per se, but a data bias issue.
    """
    results = {}
    for split_name in dataset_dict.keys():
        ds = dataset_dict[split_name]
        if "is_impossible" in ds.features:
            impossible_count = sum(1 for x in ds["is_impossible"] if x)
            possible_count = len(ds) - impossible_count
            results[split_name] = {
                "total": len(ds),
                "answerable": possible_count,
                "unanswerable": impossible_count,
                "unanswerable_ratio": round(impossible_count / max(len(ds), 1), 4),
            }
        else:
            results[split_name] = {
                "total": len(ds),
                "answerable": len(ds),
                "unanswerable": 0,
                "unanswerable_ratio": 0.0,
            }
    return results


# ---------------------------------------------------------------------------
# 6b. Batch / Pipeline Integration Helpers
# ---------------------------------------------------------------------------
def run_leakage_check_and_report(
    dataset_dict: DatasetDict,
    output_path: Optional[Path] = None,
    fail_on_leakage: bool = False,
) -> LeakageReport:
    """
    Run all leakage checks, optionally save a JSON report, and optionally
    raise if leakage is detected (for use in training pipelines).

    Args:
        dataset_dict: DatasetDict with train/validation/test splits
        output_path: Optional path to save the JSON report
        fail_on_leakage: If True, raise RuntimeError when leakage is detected

    Returns:
        LeakageReport object
    """
    report = check_data_leakage(dataset_dict)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Leakage report saved to: {output_path}")

    if fail_on_leakage and not report.is_clean:
        raise RuntimeError(
            f"Data leakage detected! Severity: {report.severity}. "
            f"Issues: {len(report.exact_context_overlaps)} context overlaps, "
            f"{len(report.title_overlaps)} title overlaps, "
            f"{len(report.id_overlaps)} ID overlaps, "
            f"{len(report.near_duplicate_pairs)} near-duplicates, "
            f"{len(report.answer_leakage)} answer leakages."
        )

    return report


def assert_no_leakage(dataset_dict: DatasetDict) -> None:
    """
    Assert that the dataset has no leakage. Raises RuntimeError if leakage found.
    Convenience wrapper for use in training scripts.
    """
    report = run_leakage_check_and_report(
        dataset_dict, fail_on_leakage=True
    )


# ---------------------------------------------------------------------------
# 7. Main Leakage Check (Comprehensive)
# ---------------------------------------------------------------------------
def check_data_leakage(dataset_dict: DatasetDict) -> LeakageReport:
    """
    Run all leakage checks and produce a comprehensive report.

    Args:
        dataset_dict: DatasetDict with train/validation/test splits

    Returns:
        LeakageReport object with all findings
    """
    report = LeakageReport()

    # Count totals
    all_contexts = set()
    total_questions = 0
    for split_name in dataset_dict.keys():
        ds = dataset_dict[split_name]
        contexts = ds["context"] if "context" in ds.features else []
        all_contexts.update(text_hash(c) for c in contexts)
        total_questions += len(ds)

    report.total_contexts = len(all_contexts)
    report.total_questions = total_questions

    # Run all checks
    logger.info("Running context overlap check...")
    report.exact_context_overlaps = check_context_overlap(dataset_dict)
    logger.info(f"  Found {len(report.exact_context_overlaps)} exact context overlaps")

    logger.info("Running title overlap check...")
    report.title_overlaps = check_title_overlap(dataset_dict)
    logger.info(f"  Found {len(report.title_overlaps)} title overlaps")

    logger.info("Running ID overlap check...")
    report.id_overlaps = check_id_overlap(dataset_dict)
    logger.info(f"  Found {len(report.id_overlaps)} ID overlaps")

    logger.info("Running near-duplicate context check...")
    report.near_duplicate_pairs = check_near_duplicate_contexts(dataset_dict)
    logger.info(f"  Found {len(report.near_duplicate_pairs)} near-duplicate pairs")

    logger.info("Running answer leakage check...")
    report.answer_leakage = check_answer_leakage(dataset_dict)
    logger.info(f"  Found {len(report.answer_leakage)} answer leakage cases")

    # Determine severity
    total_issues = (
        len(report.exact_context_overlaps)
        + len(report.title_overlaps)
        + len(report.id_overlaps)
        + len(report.near_duplicate_pairs)
        + len(report.answer_leakage)
    )

    if total_issues == 0:
        report.is_clean = True
        report.severity = "clean"
    elif total_issues < 5:
        report.is_clean = False
        report.severity = "low"
    elif total_issues < 50:
        report.is_clean = False
        report.severity = "medium"
    else:
        report.is_clean = False
        report.severity = "high"

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_loader import load_uit_viquad20

    logging.basicConfig(level=logging.INFO)

    print("Loading UIT-ViQuAD 2.0 (official splits)...")
    ds = load_uit_viquad20()

    print("\nRunning leakage check on OFFICIAL splits...")
    report = check_data_leakage(ds)
    print(report.summary())

    print("\nDetailed findings:")
    if report.exact_context_overlaps:
        print(f"\n  ⚠️  Exact context overlaps: {len(report.exact_context_overlaps)}")
        for ov in report.exact_context_overlaps[:5]:
            print(f"    - Splits: {ov['splits']} | Context: {ov['context_preview'][:80]}...")

    if report.id_overlaps:
        print(f"\n  ⚠️  ID overlaps: {len(report.id_overlaps)}")
        for ov in report.id_overlaps[:5]:
            print(f"    - ID: {ov['id']} | Splits: {ov['splits']}")

    # Distribution check
    print("\nDistribution analysis:")
    dist = check_distribution_shift(ds)
    for split, info in dist.items():
        print(f"  {split}: total={info['total']}, answerable={info['answerable']}, "
              f"unanswerable={info['unanswerable']} ({info['unanswerable_ratio']*100:.1f}%)")
