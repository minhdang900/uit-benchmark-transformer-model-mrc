# Results

**Status: not yet measured.**

This directory holds machine-generated evaluation output. Nothing is written here
by hand.

Expected contents once the benchmark has been run:

| File | Produced by |
|---|---|
| `benchmark_summary.json` | `src/train_mrc_transformer.py` (benchmark mode) |
| `<model>/training_results.json` | `src/train_mrc_transformer.py` (per model) |
| `baseline_bm25.json` | `src/baseline_bm25.py` |
| `stress_test_by_category.json` | `src/error_analysis.py` |

## Rule

No number appears in the README, the report, or the slides unless it can be
traced to a committed file in this directory.
