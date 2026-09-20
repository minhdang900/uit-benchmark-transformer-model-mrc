"""Điểm các tập con SLICE của stress-test v2, tính từ dự đoán validation có sẵn.

    python scripts/stress_v2_slice_scores.py

Các mục slice (E1a, E1b, E2a, E2b, E3a, E4) là câu validation nguyên văn, nên
điểm của chúng rút ra được từ ``results/predictions_<run>_validation.json`` mà
không cần chạy lại mô hình. Các mục perturbation (E2c, E3b, E5) có context hoặc
câu hỏi mới — chúng CẦN ``scripts/run_eval.py --dataset stress2``.

Ghi ra ``results/stress_v2_slice_scores.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file  # noqa: E402
from mrc.evaluate import _git_commit  # noqa: E402
from mrc.metrics import evaluate  # noqa: E402
from mrc.stress_v2 import load_stress_v2, score_report  # noqa: E402

DEFAULT_RUNS = ["abstain", "baseline", "phobert_tuned", "phobert_seed13_tuned",
                "xlmr_seed13_tuned", "xlmr_256_tuned"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/stress_v2_slice_scores.json")
    args = ap.parse_args(argv)

    examples, meta = load_stress_v2("data/stress_test_v2/stress_v2.json")
    slices = [e for e in examples if meta[e.qid]["role"] == "slice"]
    validation = {e.qid: list(e.answers) for e in load_squad_file("data/raw/viquad2_validation.json")}

    out = {"commit": _git_commit(), "note": "chỉ các tập con slice; τ theo tên lần chạy", "runs": {}}
    for run in args.runs:
        path = Path(args.results) / f"predictions_{run}_validation.json"
        if not path.exists():
            print(f"bỏ qua {run}: không có {path}")
            continue
        preds = json.loads(path.read_text(encoding="utf-8"))
        preds = preds.get("predictions", preds)
        mapped = {e.qid: preds[meta[e.qid]["source_qid"]] for e in slices}
        report = score_report(mapped, slices, {q: meta[q] for q in mapped})
        full = evaluate(preds, validation)
        out["runs"][run] = {
            "validation_EM": round(full["EM"], 2),
            "validation_EM_answerable": round(full["EM_answerable"], 2),
            "by_subset": {k: {"EM": v["EM"], "F1": v["F1"], "n": v["n"]}
                          for k, v in report["by_subset"].items()},
        }
        row = "  ".join(f"{k} {v['EM']:5.1f}" for k, v in report["by_subset"].items())
        print(f"{run:22s} val {full['EM']:5.1f} | {row}")

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
