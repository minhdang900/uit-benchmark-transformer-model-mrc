# -*- coding: utf-8 -*-
"""
Baseline Model: BM25 / TF-IDF for Vietnamese Extractive QA
===========================================================
Course: CS221 — Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

Implements sparse retrieval-based QA baselines:
  1. BM25 (Best Matching 25) — probabilistic ranking
  2. TF-IDF + Cosine Similarity
  3. Hybrid BM25+TF-IDF ensemble

Usage:
    python src/baseline_bm25.py --evaluate --dataset taidng/UIT-ViQuAD2.0
    python src/baseline_bm25.py --single --context "..." --question "..."

Author: chu trach CS221 (NLP Researcher)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import load_uit_viquad20, context_level_split
from eval_metrics import normalize_text, compute_exact_match, compute_f1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 Implementation
# ---------------------------------------------------------------------------
class BM25Retriever:
    """
    BM25 ranking function for Vietnamese QA.

    BM25 score = sum over query terms of:
        IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))

    Where:
      - f(q_i, D) = term frequency of query term q_i in document D
      - |D| = document length (number of tokens)
      - avgdl = average document length
      - k1 = term frequency saturation parameter (default: 1.5)
      - b = length normalization parameter (default: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.doc_len: Dict[int, int] = {}
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.doc_tokens: Dict[int, List[str]] = {}
        self.doc_ids: List[int] = []
        self.corpus: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple Vietnamese tokenizer (word-level splitting)."""
        # Lowercase and normalize
        tokens = normalize_text(text).split()
        return tokens

    def fit(self, corpus: List[str]) -> "BM25Retriever":
        """Index a corpus of documents."""
        self.corpus = corpus
        total_tokens = 0

        for idx, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            self.doc_tokens[idx] = tokens
            self.doc_len[idx] = len(tokens)
            total_tokens += len(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

            self.doc_ids.append(idx)

        self.avgdl = total_tokens / len(corpus) if corpus else 1.0

        # Compute IDF scores
        N = len(corpus)
        for word, freq in self.doc_freqs.items():
            idf_score = math.log(N - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf_score

        return self

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query against a single document."""
        tokens = self._tokenize(query)
        score = 0.0
        doc_tokens = self.doc_tokens.get(doc_idx, [])
        term_freqs = Counter(doc_tokens)

        doc_length = self.doc_len[doc_idx]

        for token in tokens:
            if token not in self.idf:
                continue
            tf = term_freqs.get(token, 0)
            idf = self.idf[token]

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
            score += idf * (numerator / denominator if denominator > 0 else 0)

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Rank documents by BM25 score for a query."""
        scores = []
        for idx in self.doc_ids:
            score = self.score(query, idx)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# TF-IDF Retriever
# ---------------------------------------------------------------------------
class TFIDFRetriever:
    """TF-IDF + Cosine Similarity retriever."""

    def __init__(self):
        self.doc_vectors: Dict[int, Dict[str, float]] = {}
        self.doc_len: Dict[int, int] = {}
        self.doc_ids: List[int] = []
        self.corpus: List[str] = []
        self.idf_scores: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return normalize_text(text).split()

    def fit(self, corpus: List[str]) -> "TFIDFRetriever":
        """Index a corpus using TF-IDF."""
        self.corpus = corpus
        N = len(corpus)
        doc_freqs: Dict[str, int] = defaultdict(int)
        doc_tokens: Dict[int, List[str]] = {}

        for idx, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            doc_tokens[idx] = tokens
            self.doc_len[idx] = len(tokens)
            self.doc_ids.append(idx)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freqs[token] += 1

        # Compute IDF
        for word, df in doc_freqs.items():
            self.idf_scores[word] = math.log((N + 1) / (df + 1)) + 1

        # Compute TF-IDF vectors
        for idx in self.doc_ids:
            tokens = doc_tokens[idx]
            tf = Counter(tokens)
            total = len(tokens)
            self.doc_vectors[idx] = {}
            for token, count in tf.items():
                tf_norm = count / total if total > 0 else 0
                self.doc_vectors[idx][token] = tf_norm * self.idf_scores.get(token, 0)

        return self

    def _cosine_similarity(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute cosine similarity between query and document."""
        query_tf = Counter(query_tokens)
        query_magnitude = 0.0
        dot_product = 0.0

        doc_vector = self.doc_vectors.get(doc_idx, {})

        for token, count in query_tf.items():
            if token in doc_vector:
                idf = self.idf_scores.get(token, 0)
                query_weight = (count / len(query_tokens)) * idf if query_tokens else 0
                dot_product += query_weight * doc_vector[token]
                query_magnitude += query_weight ** 2

        doc_magnitude = math.sqrt(sum(w ** 2 for w in doc_vector.values()))
        query_magnitude = math.sqrt(query_magnitude)

        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0

        return dot_product / (query_magnitude * doc_magnitude)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Rank documents by cosine similarity."""
        query_tokens = self._tokenize(query)
        scores = []
        for idx in self.doc_ids:
            score = self._cosine_similarity(query_tokens, idx)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# QA Model (Baseline)
# ---------------------------------------------------------------------------
class BaselineBM25QA:
    """
    Extractive QA baseline using BM25 sentence-level retrieval.

    For a given (context, question), splits the context into sentences,
    retrieves the most relevant sentence using BM25, and extracts
    the answer via keyword matching.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25Retriever(k1=k1, b=b)
        self.tfidf = TFIDFRetriever()
        self.fitted = False

    @staticmethod
    def _split_sentences(context: str) -> List[str]:
        """Split Vietnamese text into sentences."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+|\n', context)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    @staticmethod
    def _extract_answer_span(sentence: str, question: str, context: str) -> str:
        """
        Extract answer from the retrieved sentence using keyword overlap.
        This is a heuristic — the real BM25 baseline would use the question
        words to find the most likely answer span in the sentence.
        """
        question_words = set(normalize_text(question).split())
        sentence_words = sentence.split()

        # Find keyword positions
        keyword_positions = []
        for i, word in enumerate(sentence_words):
            if normalize_text(word) in question_words:
                keyword_positions.append(i)

        if not keyword_positions:
            # Fallback: return first few words
            return " ".join(sentence_words[:10])

        # Return words around the first keyword (±5 words)
        start = max(0, keyword_positions[0] - 3)
        end = min(len(sentence_words), keyword_positions[-1] + 5)
        return " ".join(sentence_words[start:end])

    def fit_context(self, context: str) -> None:
        """Fit the retriever on a single context (for single-question prediction)."""
        sentences = self._split_sentences(context)
        self.bm25 = BM25Retriever(k1=1.5, b=0.75)
        self.bm25.fit(sentences)
        self.tfidf = TFIDFRetriever()
        self.tfidf.fit(sentences)
        self._current_context = context
        self.fitted = True

    def predict(self, context: str, question: str, top_k: int = 3) -> str:
        """Predict answer for a single (context, question) pair."""
        # Fit on the current context
        self.fit_context(context)

        # Retrieve top-k sentences using BM25
        bm25_results = self.bm25.search(question, top_k=top_k)

        if not bm25_results or bm25_results[0][1] < 0.01:
            # Fallback to TF-IDF
            tfidf_results = self.tfidf.search(question, top_k=top_k)
            if tfidf_results and tfidf_results[0][1] > 0:
                best_idx = tfidf_results[0][0]
            else:
                # Return empty or first sentence
                sentences = self._split_sentences(context)
                return "" if not sentences else self._extract_answer_span(sentences[0], question, context)
        else:
            best_idx = bm25_results[0][0]

        # Get the best sentence
        sentences = self._split_sentences(context)
        best_sentence = sentences[best_idx]

        # Extract answer span from the sentence
        answer = self._extract_answer_span(best_sentence, question, context)

        return answer

    def evaluate(self, dataset, top_k: int = 3) -> Dict[str, float]:
        """
        Evaluate on a SQuAD-format dataset.

        Args:
            dataset: Dataset with 'context', 'question', 'answers' columns
            top_k: Number of top sentences to consider

        Returns:
            Dict with {'exact_match': ..., 'f1': ..., 'total': ...}
        """
        total_em = 0.0
        total_f1 = 0.0
        count = 0

        for example in dataset:
            context = example["context"]
            question = example["question"]
            ground_truths = example["answers"]["text"]

            prediction = self.predict(context, question, top_k=top_k)

            if not ground_truths:
                continue

            em_scores = [compute_exact_match(prediction, gt) for gt in ground_truths]
            f1_scores = [compute_f1(prediction, gt) for gt in ground_truths]

            total_em += max(em_scores)
            total_f1 += max(f1_scores)
            count += 1

        em_score = 100.0 * total_em / count if count > 0 else 0.0
        f1_score = 100.0 * total_f1 / count if count > 0 else 0.0

        return {
            "exact_match": round(em_score, 2),
            "f1": round(f1_score, 2),
            "total": count,
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BM25/TF-IDF Baseline for Vietnamese Extractive QA"
    )
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on UIT-ViQuAD 2.0")
    parser.add_argument("--dataset", type=str, default="taidng/UIT-ViQuAD2.0",
                        help="HuggingFace dataset ID")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate on")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top sentences to retrieve")
    parser.add_argument("--single", action="store_true",
                        help="Run single prediction mode")
    parser.add_argument("--context", type=str, default=None,
                        help="Context text for single mode")
    parser.add_argument("--question", type=str, default=None,
                        help="Question text for single mode")
    parser.add_argument("--no_context_split", action="store_true",
                        help="Use official splits instead of context-level split")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                        help="Output results path")
    parser.add_argument("--subset_size", type=int, default=500,
                        help="Limit evaluation to first N examples for speed")

    args = parser.parse_args()

    if args.single:
        if not args.context or not args.question:
            print("Error: --context and --question required for --single mode")
            return

        model = BaselineBM25QA()
        answer = model.predict(args.context, args.question)
        print(f"\nContext: {args.context[:200]}...")
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
        return

    if args.evaluate:
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Loading dataset: {args.dataset}")
        raw = load_uit_viquad20()

        if not args.no_context_split:
            logger.info("Applying context-level split...")
            raw = context_level_split(raw)

        data = raw[args.split]
        if args.subset_size:
            data = data.select(range(min(args.subset_size, len(data))))
            logger.info(f"Using subset of {len(data)} examples")

        logger.info(f"Evaluating baseline BM25 on {len(data)} examples...")

        model = BaselineBM25QA()
        results = model.evaluate(data, top_k=args.top_k)

        print(f"\n{'='*60}")
        print(f"BM25 Baseline Results ({args.split} split)")
        print(f"{'='*60}")
        print(f"  Total examples:  {results['total']}")
        print(f"  Exact Match:     {results['exact_match']:.2f}%")
        print(f"  F1-Score:        {results['f1']:.2f}%")
        print(f"  Top-k:           {args.top_k}")
        print(f"{'='*60}")

        # Save results
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "model": "BM25-baseline",
                "dataset": args.dataset,
                "split": args.split,
                "top_k": args.top_k,
                "context_level_split": not args.no_context_split,
                "num_examples": results["total"],
                "exact_match": results["exact_match"],
                "f1": results["f1"],
            }, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
