"""Chọn ngưỡng "không có đáp án" τ cho từng hệ thống trên DEV, áp MỘT lần lên validation.

Đầu vào: ``results/windows_<run>_{dev,validation}.json`` (``scripts/score_windows.py``).
Đầu ra: ``results/thresholds.json`` gồm, cho mỗi hệ thống,

* τ chọn trên dev (F1 lớn nhất, hoà thì gần 0 nhất),
* điểm validation ở τ = 0 và ở τ đã chọn — tách câu có / không có đáp án, tỉ lệ từ chối,
* đường precision/recall của lớp "không có đáp án" theo τ, đo trên DEV (mô tả, không chọn gì),

và cho từng cặp hệ thống: EM trên các câu có đáp án mà CẢ HAI đều trả lời —
thước đo "đọc" không bị lẫn với "sẵn lòng trả lời".

    python scripts/calibrate_thresholds.py --runs xlmr phobert
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file, references_from, split_by_context  # noqa: E402
from mrc.metrics import evaluate, exact_match, metric_max_over_ground_truths  # noqa: E402
from mrc.threshold import best_threshold, predictions_at  # noqa: E402


def load_windows(run: str, split: str) -> dict:
    return json.loads((Path("results") / f"windows_{run}_{split}.json").read_text())["records"]


def scores(preds: dict, refs: dict) -> dict:
    m = evaluate(preds, refs)
    ans = [q for q in refs if refs[q]]
    imp = [q for q in refs if not refs[q]]
    n = len(preds)
    return {
        "EM": round(m["EM"], 4), "F1": round(m["F1"], 4),
        "EM_answerable": round(100 * sum(metric_max_over_ground_truths(exact_match, preds[q], refs[q])
                                         for q in ans) / len(ans), 4),
        "EM_impossible": round(100 * sum(1 for q in imp if not preds[q]) / len(imp), 4),
        "abstain_rate": round(100 * sum(1 for p in preds.values() if not p) / n, 4),
        "n": n,
    }


def pr_curve(records: dict, refs: dict, n_points: int = 41) -> list[dict]:
    deltas = sorted(w["delta"] for ws in records.values() for w in ws if w["delta"] is not None)
    lo, hi = deltas[int(0.02 * len(deltas))], deltas[int(0.98 * len(deltas)) - 1]
    out = []
    for k in range(n_points):
        tau = -(lo + (hi - lo) * k / (n_points - 1))
        preds = predictions_at(records, tau)
        tp = sum(1 for q in refs if not refs[q] and not preds[q])
        fp = sum(1 for q in refs if refs[q] and not preds[q])
        fn = sum(1 for q in refs if not refs[q] and preds[q])
        out.append({"tau": round(tau, 4),
                    "precision": round(100 * tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(100 * tp / (tp + fn), 3) if tp + fn else None})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data/raw")
    args = ap.parse_args(argv)

    train = load_squad_file(Path(args.data_dir) / "viquad2_train.json")
    _, dev = split_by_context(train, val_frac=0.05, seed=42)
    dev_refs = references_from(dev)
    val_refs = references_from(load_squad_file(Path(args.data_dir) / "viquad2_validation.json"))

    systems, val_preds = {}, {}
    for run in args.runs:
        dev_w, val_w = load_windows(run, "dev"), load_windows(run, "validation")
        chosen = best_threshold(dev_w, dev_refs)
        tau = chosen["tau"]
        val_preds[run] = predictions_at(val_w, tau)
        # Dự đoán ở τ đã chọn — cho diagnose.py so sánh cặp ở τ hiệu chỉnh.
        (Path("results") / f"predictions_{run}_tuned_validation.json").write_text(
            json.dumps(val_preds[run], ensure_ascii=False, indent=0), encoding="utf-8")
        systems[run] = {
            "tau": tau,
            "dev_at_tau0": scores(predictions_at(dev_w, 0.0), dev_refs),
            "dev_at_tau": scores(predictions_at(dev_w, tau), dev_refs),
            "validation_at_tau0": scores(predictions_at(val_w, 0.0), val_refs),
            "validation_at_tau": scores(val_preds[run], val_refs),
            "dev_noanswer_pr_curve": pr_curve(dev_w, dev_refs),
        }
        print(run, "tau", tau, "val@0", systems[run]["validation_at_tau0"],
              "val@tau", systems[run]["validation_at_tau"])

    pairs = {}
    ans = [q for q in val_refs if val_refs[q]]
    for a, b in itertools.combinations(args.runs, 2):
        for label, pa, pb in (
            ("tau0", predictions_at(load_windows(a, "validation"), 0.0),
             predictions_at(load_windows(b, "validation"), 0.0)),
            ("tuned", val_preds[a], val_preds[b]),
        ):
            both = [q for q in ans if pa[q] and pb[q]]
            em = lambda p: round(100 * sum(metric_max_over_ground_truths(exact_match, p[q], val_refs[q])
                                           for q in both) / len(both), 4)
            pairs[f"{a}__{b}__{label}"] = {"n_both_answered": len(both), a: em(pa), b: em(pb)}
            print(a, b, label, pairs[f"{a}__{b}__{label}"])

    Path("results/thresholds.json").write_text(json.dumps(
        {"selection": "tau maximises F1 on dev (5% train contexts, split seed 42); "
                      "validation scored once at chosen tau",
         "systems": systems, "both_answered": pairs}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
