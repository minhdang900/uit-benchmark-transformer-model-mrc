# -*- coding: utf-8 -*-
"""
Fine-tuning Script for Vietnamese Extractive MRC (Transformer Models)
=====================================================================
Course: CS221 — Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

Fine-tunes and benchmarks multiple Transformer models on UIT-ViQuAD 2.0:

  - vinai/phobert-base-v2   (Monolingual, 12 layers)
  - vinai/phobert-large      (Monolingual, 24 layers)
  - FPTAI/videberta-base     (Monolingual, 12 layers)
  - bert-base-multilingual-cased (Multilingual, 12 layers)
  - xlm-roberta-base         (Multilingual, 12 layers)

Supports:
  - Context-level split (anti-leakage)
  - Distributed training (accelerate)
  - Logging with Weights & Biases (optional)
  - Evaluation metrics: EM (Exact Match) and F1-Score

Usage:
    python src/train_mrc_transformer.py --model_name vinai/phobert-base-v2 --epochs 5
    python src/train_mrc_transformer.py --model_name vinai/phobert-base-v2 --eval_only --checkpoint ./checkpoints/phobert
    python src/train_mrc_transformer.py --benchmark  # Run all models

Author: chu trach CS221 (NLP Researcher)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qa_postprocessing import compute_span_metrics, postprocess_qa_predictions
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import (
    DEFAULT_DATASET_ID,
    DEFAULT_MAX_LENGTH,
    DEFAULT_STRIDE,
    SUPPORTED_MODELS,
    context_level_split,
    load_multi_source_datasets,
    load_uit_viquad20,
    tokenize_dataset,
)

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import DatasetDict

# Try to import evaluate library for metrics
try:
    import evaluate
    SQUAD_METRIC = evaluate.load("squad_v2")
except Exception:
    SQUAD_METRIC = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "vinai/phobert-base-v2": {
        "batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 5,
        "max_length": 256,  # PhoBERT position-embedding limit is 258
        "stride": 128,
        "weight_decay": 0.01,
    },
    "vinai/phobert-large": {
        "batch_size": 8,
        "learning_rate": 2e-5,
        "epochs": 5,
        "max_length": 256,  # PhoBERT position-embedding limit is 258
        "stride": 128,
        "weight_decay": 0.01,
    },
    "FPTAI/videberta-base": {
        "batch_size": 16,
        "learning_rate": 3e-5,
        "epochs": 5,
        "max_length": 384,
        "stride": 128,
        "weight_decay": 0.01,
    },
    "bert-base-multilingual-cased": {
        "batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 5,
        "max_length": 384,
        "stride": 128,
        "weight_decay": 0.01,
    },
    "xlm-roberta-base": {
        "batch_size": 16,
        "learning_rate": 3e-5,
        "epochs": 5,
        "max_length": 384,
        "stride": 128,
        "weight_decay": 0.01,
    },
}



# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------
def load_and_prepare_datasets(
    model_name: str,
    use_context_split: bool = True,
    include_stress_test: bool = True,
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
    cache_dir: Optional[str] = None,
) -> Tuple[DatasetDict, AutoTokenizer]:
    """
    Load, split, and tokenize the dataset for the given model.
    """
    logger.info(f"Preparing datasets for model: {model_name}")

    # Load raw data
    raw_datasets = load_uit_viquad20()

    # Apply context-level split
    if use_context_split:
        logger.info("Applying context-level split (anti-leakage)...")
        raw_datasets = context_level_split(raw_datasets, random_seed=42)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)

    # Add special tokens if needed
    if model_name == "vinai/phobert-base-v2":
        # PhoBERT uses RoBERTa-style tokenizer; ensure </s> exists for QA
        if tokenizer.sep_token is None:
            tokenizer.sep_token = tokenizer.eos_token

    # Tokenize
    logger.info("Tokenizing datasets...")
    tokenized_datasets = DatasetDict()
    for split_name in ["train", "validation", "test"]:
        tokenized_datasets[split_name] = tokenize_dataset(
            raw_datasets[split_name], tokenizer, max_length, stride
        )
        logger.info(f"  {split_name}: {len(tokenized_datasets[split_name])} tokenized features")

    return tokenized_datasets, tokenizer


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute EM (Exact Match) and F1-Score for extractive QA.
    Uses the official SQuAD v2 evaluation (supports unanswerable questions).
    """
    raise NotImplementedError(
        "Span-level metrics are computed after decoding, not from raw logits.\n"
        "Use qa_postprocessing.postprocess_qa_predictions() to turn logits into\n"
        "text spans, then score with compute_span_metrics() below.\n"
        "The previous implementation compared literal f\"span_{start}_{end}\"\n"
        "strings and never measured Exact Match or F1."
    )




def train_single_model(
    model_name: str,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: Optional[str] = None,
    use_context_split: bool = True,
    include_stress_test: bool = True,
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    logging_steps: int = 50,
    eval_steps: int = 500,
    seed: int = 42,
    no_cuda: bool = False,
    cache_dir: Optional[str] = None,
    hub_push: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fine-tune a single model on UIT-ViQuAD 2.0.

    Returns a dictionary with training results and metrics.
    """
    # Load model config
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

    # Do NOT override max_position_embeddings. It is a property of the
    # pretrained checkpoint, not a runtime knob: PhoBERT supports 258 and
    # forcing 384 either errors or silently uses untrained position embeddings.
    # Cap the tokenizer instead (see MODEL_CONFIGS max_length).

    # Load tokenizer and datasets
    tokenized_datasets, tokenizer = load_and_prepare_datasets(
        model_name=model_name,
        use_context_split=use_context_split,
        include_stress_test=include_stress_test,
        max_length=max_length,
        stride=stride,
        cache_dir=cache_dir,
    )

    # Load model
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
    )

    # Set up output directory
    if output_dir is None:
        model_safe_name = model_name.replace("/", "_")
        output_dir = f"./checkpoints/{model_safe_name}"

    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        fp16=not no_cuda and torch.cuda.is_available(),
        report_to=["none"],  # Disable W&B by default
        run_name=f"{model_name.replace('/', '_')}_mrc",
        seed=seed,
        dataloader_num_workers=os.cpu_count() or 0,
        remove_unused_columns=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info(f"Starting training: {model_name} for {num_epochs} epochs...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_result = trainer.evaluate(tokenized_datasets["test"])

    # Save results
    results = {
        "model_name": model_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_loss": train_result.metrics.get("train_loss"),
        "test_exact_match": test_result.get("eval_exact", test_result.get("eval_exact_match", 0)),
        "test_f1": test_result.get("eval_f1", 0),
        "training_time_seconds": round(training_time, 2),
        "training_time_minutes": round(training_time / 60, 2),
        "use_context_split": use_context_split,
        "include_stress_test": include_stress_test,
        "max_length": max_length,
        "output_dir": output_dir,
    }

    # Save results to JSON
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Test EM: {results['test_exact_match']:.2f}%, Test F1: {results['test_f1']:.2f}%")

    return results


def benchmark_all_models(
    output_dir: str = "./benchmark_results",
    use_context_split: bool = True,
    quick_mode: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Run benchmark across all supported models.

    Args:
        output_dir: Directory to save benchmark results
        use_context_split: Whether to use context-level anti-leakage split
        quick_mode: If True, use fewer epochs for quick comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results: Dict[str, Dict[str, Any]] = {}

    for model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name].copy()
        if quick_mode:
            config["epochs"] = 2  # Reduced for quick benchmarking

        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {model_name}")
        logger.info(f"Config: {config}")
        logger.info(f"{'='*60}")

        try:
            model_output_dir = os.path.join(output_dir, model_name.replace("/", "_"))
            results = train_single_model(
                model_name=model_name,
                output_dir=model_output_dir,
                use_context_split=use_context_split,
                include_stress_test=True,
                **config,
            )
            all_results[model_name] = results
        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}")
            all_results[model_name] = {"error": str(e), "model_name": model_name}

    # Save benchmark summary
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nBenchmark summary saved to: {summary_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (Comparison Table)")
    print("=" * 80)
    print(f"{'Model':<40} {'EM (%)':>10} {'F1 (%)':>10} {'Time (min)':>12} {'Split':>8}")
    print("-" * 80)
    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name:<40} {'ERROR':>10} {'':>10} {'':>12} {'':>8}")
        else:
            print(
                f"{model_name:<40} "
                f"{results['test_exact_match']:>10.2f} "
                f"{results['test_f1']:>10.2f} "
                f"{results['training_time_minutes']:>12.2f} "
                f"{'ctx-split' if results['use_context_split'] else 'official':>8}"
            )
    print("=" * 80)

    return all_results


def eval_single_model(
    model_name: str,
    checkpoint_dir: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
    use_context_split: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a pre-trained model checkpoint on the test set.
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint_dir)

    # Load and prepare data
    raw_datasets = load_uit_viquad20()
    if use_context_split:
        raw_datasets = context_level_split(raw_datasets, random_seed=42)

    tokenized_datasets = DatasetDict()
    for split_name in ["train", "validation", "test"]:
        tokenized_datasets[split_name] = tokenize_dataset(
            raw_datasets[split_name], tokenizer, max_length, stride
        )

    # Evaluate
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_eval_batch_size=16,
        remove_unused_columns=False,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets["test"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate(tokenized_datasets["test"])

    return {
        "model_name": model_name,
        "checkpoint_dir": checkpoint_dir,
        "exact_match": eval_results.get("eval_exact", 0),
        "f1": eval_results.get("eval_f1", 0),
        "eval_loss": eval_results.get("eval_loss", 0),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune and benchmark Vietnamese MRC Transformer models on UIT-ViQuAD 2.0"
    )
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2",
                        help="Model name or path (e.g., vinai/phobert-base-v2)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="Stride for tokenization")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Checkpoint directory for evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation, no training")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark on all models")
    parser.add_argument("--quick", action="store_true",
                        help="Quick benchmark mode (2 epochs)")
    parser.add_argument("--no_context_split", action="store_true",
                        help="Use official splits instead of context-level split")
    parser.add_argument("--no_stress_test", action="store_true",
                        help="Exclude stress-test data")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--hub_push", type=str, default=None,
                        help="Push to HuggingFace Hub with this repo name")

    args = parser.parse_args()

    use_context_split = not args.no_context_split
    include_stress_test = not args.no_stress_test

    if args.benchmark:
        # Run all models
        results = benchmark_all_models(
            output_dir=args.output_dir or "./benchmark_results",
            use_context_split=use_context_split,
            quick_mode=args.quick,
        )
        print(f"\nBenchmark complete. Results saved to ./benchmark_results/")
        return

    if args.eval_only:
        if not args.checkpoint_dir:
            logger.error("--checkpoint_dir is required for --eval_only")
            return
        results = eval_single_model(
            model_name=args.model_name,
            checkpoint_dir=args.checkpoint_dir,
            max_length=args.max_length,
            stride=args.stride,
            use_context_split=use_context_split,
        )
        print(f"\nEvaluation results for {args.model_name}:")
        print(f"  Exact Match: {results['exact_match']:.2f}%")
        print(f"  F1-Score: {results['f1']:.2f}%")
        return

    # Single model training
    config = MODEL_CONFIGS.get(args.model_name, {})
    results = train_single_model(
        model_name=args.model_name,
        num_epochs=args.epochs if args.epochs != 5 else config.get("epochs", 5),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        use_context_split=use_context_split,
        include_stress_test=include_stress_test,
        max_length=args.max_length,
        stride=args.stride,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        no_cuda=args.no_cuda,
        cache_dir=args.cache_dir,
        hub_push=args.hub_push,
    )

    print(f"\nTraining complete for {args.model_name}:")
    print(f"  Train Loss: {results.get('train_loss', 'N/A')}")
    print(f"  Test EM: {results['test_exact_match']:.2f}%")
    print(f"  Test F1: {results['test_f1']:.2f}%")
    print(f"  Training Time: {results['training_time_minutes']:.2f} minutes")


if __name__ == "__main__":
    main()
