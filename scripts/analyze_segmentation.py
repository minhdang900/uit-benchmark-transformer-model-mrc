"""Đo tương thích biên đáp án vàng ↔ biên từ (pyvi) trên TOÀN BỘ một split.

Không cần model. Ghi ``results/segmentation_<split>.json`` gồm tổng hợp và nhãn
từng câu (``aligned``, ``oracle_em``...), để ``scripts/diagnose.py`` ghép với dự
đoán của từng model.

    python scripts/analyze_segmentation.py --split validation
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file  # noqa: E402
from mrc.evaluate import _git_commit  # noqa: E402
from mrc.segmentation import boundary_alignment  # noqa: E402
from mrc.segmented_tokenizer import segment_with_offsets  # noqa: E402


def coverage(context: str, words) -> float:
    """Tỉ lệ ký tự không-trắng của context được một từ đã căn phủ tới."""
    covered = sum(e - s - context[s:e].count(" ") for _, s, e in words)
    total = sum(1 for c in context if not c.isspace())
    return covered / total if total else 1.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    examples = load_squad_file(Path(args.data_dir) / f"viquad2_{args.split}.json")
    t0 = time.time()
    words_by_ctx = {c: segment_with_offsets(c) for c in {e.context for e in examples}}
    seg_seconds = time.time() - t0

    items: dict[str, dict] = {}
    for ex in examples:
        if ex.is_impossible:
            continue
        r = boundary_alignment(ex.context, ex.answers, ex.answer_start, words_by_ctx[ex.context])
        if r is None:
            items[ex.qid] = {"located": False}
            continue
        items[ex.qid] = {"located": True, **r}

    located = [v for v in items.values() if v["located"]]
    n = len(located)
    mis = [v for v in located if not v["aligned"]]
    covs = [coverage(c, w) for c, w in words_by_ctx.items()]
    summary = {
        "split": args.split,
        "segmenter": "pyvi.ViTokenizer",
        "n_answerable": len(items),
        "n_located": n,
        "n_aligned": n - len(mis),
        "n_misaligned": len(mis),
        "misaligned_pct": round(100 * len(mis) / n, 2) if n else 0.0,
        "misaligned_start_cut": sum(v["start_cut"] for v in mis),
        "misaligned_end_cut": sum(v["end_cut"] for v in mis),
        "oracle_em_all_pct": round(100 * sum(v["oracle_em"] for v in located) / n, 2),
        "oracle_em_misaligned_pct": round(100 * sum(v["oracle_em"] for v in mis) / len(mis), 2)
        if mis else None,
        "n_contexts": len(words_by_ctx),
        "segmentation_seconds": round(seg_seconds, 1),
        "char_coverage_min": round(min(covs), 4),
        "char_coverage_mean": round(statistics.mean(covs), 4),
        "contexts_with_coverage_below_99pct": sum(c < 0.99 for c in covs),
    }
    by_qid = {e.qid: e for e in examples}
    examples_mis = [
        {"qid": q, "gold": by_qid[q].answers, "oracle_span": v["oracle_span"]}
        for q, v in items.items() if v.get("located") and not v["aligned"]
    ][:25]

    out = Path(args.out or f"results/segmentation_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "commit": _git_commit(),
        "summary": summary,
        "misaligned_examples": examples_mis,
        "items": items,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
