"""Chấm bộ stress-test v2 ở ngưỡng τ đã chọn TRÊN DEV, từ điểm từng cửa sổ.

    python scripts/score_stress2_tuned.py --runs xlmr xlmr_seed13 phobert phobert_seed13 xlmr_256

Đọc ``results/windows_<run>_stress2.json`` (sinh bằng ``scripts/score_windows.py
--split stress2``) và ``results/thresholds.json``, rồi dựng lại dự đoán ở τ của dev
mà không chạy lại model. Ghi ``results/eval_<run>_stress2_tuned.json`` cùng lược đồ
với ``eval_<run>_stress2.json`` để báo cáo đọc được như nhau.

τ KHÔNG BAO GIỜ được chọn trên bộ stress-test: nó là bộ chẩn đoán. τ lấy nguyên từ
``thresholds.json`` (tối đa F1 trên dev = 5% context của train, split seed 42), áp
đúng một lần lên đây — cùng giao thức đã dùng cho validation.

Kiểm chứng: ở τ = 0 phải dựng lại đúng điểm của ``eval_<run>_stress2.json`` (vốn
chấm bằng ``predict()`` trực tiếp). Cờ ``--check-tau0`` in ra so sánh đó.
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

from mrc.metrics import evaluate as evaluate_metrics  # noqa: E402
from mrc.stress_v2 import load_stress_v2, score_report  # noqa: E402
from mrc.threshold import predictions_at  # noqa: E402

STRESS_V2_PATH = _ROOT / "data/stress_test_v2/stress_v2.json"


def blocks(predictions: dict[str, str], refs: dict[str, list[str]]) -> dict:
    """``overall`` / ``answerable_only`` / ``impossible_only`` — cùng dạng run_eval.py."""
    s = evaluate_metrics(predictions, refs)
    s.pop("per_item", None)
    return {
        "overall": {"EM": round(s["EM"], 4), "F1": round(s["F1"], 4), "count": s["count"]},
        "answerable_only": {"EM": round(s["EM_answerable"], 4),
                            "F1": round(s["F1_answerable"], 4), "count": s["n_answerable"]},
        "impossible_only": {"EM": round(s["EM_impossible"], 4), "count": s["n_impossible"]},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--stress", default=str(STRESS_V2_PATH), help="tệp stress-test v2 (SQuAD + khoá stress)")
    ap.add_argument("--check-tau0", action="store_true",
                    help="so điểm dựng lại ở τ=0 với eval_<run>_stress2.json")
    args = ap.parse_args(argv)

    res = Path(args.results)
    examples, meta = load_stress_v2(args.stress)
    refs = {e.qid: list(e.answers) for e in examples}
    taus = json.loads((res / "thresholds.json").read_text(encoding="utf-8"))["systems"]

    for run in args.runs:
        wpath = res / f"windows_{run}_stress2.json"
        if not wpath.exists():
            print(f"{run}: THIẾU {wpath.name} — chạy score_windows.py --split stress2 trước")
            continue
        w = json.loads(wpath.read_text(encoding="utf-8"))
        records = w["records"]
        tau = (taus.get(run) or {}).get("tau")
        if tau is None:
            print(f"{run}: THIẾU τ trong thresholds.json — bỏ qua")
            continue

        if args.check_tau0:
            direct = res / f"eval_{run}_stress2.json"
            if direct.exists():
                got = blocks(predictions_at(records, 0.0), refs)["overall"]
                want = json.loads(direct.read_text(encoding="utf-8"))["overall"]
                ok = abs(got["EM"] - want["EM"]) < 0.01 and abs(got["F1"] - want["F1"]) < 0.01
                print(f"  [τ=0] {run}: dựng lại EM {got['EM']:.2f}/F1 {got['F1']:.2f} vs "
                      f"trực tiếp EM {want['EM']:.2f}/F1 {want['F1']:.2f} "
                      f"{'KHỚP' if ok else 'LỆCH'}")

        preds = predictions_at(records, tau)
        out = {
            "model": run, "dataset": "stress-test v2", "split": "stress2",
            "tau": tau, "tau_source": "results/thresholds.json (chọn trên dev, split seed 42)",
            "checkpoint": w.get("checkpoint"), "commit": w.get("commit"),
            **blocks(preds, refs),
            "abstain_rate": round(100 * sum(1 for p in preds.values() if not p) / len(preds), 4),
            **score_report(preds, examples, meta),
        }
        path = res / f"eval_{run}_stress2_tuned.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        o = out["overall"]
        broken = {k: v["pairs"]["broken"] for k, v in out["by_subset"].items() if v.get("pairs")}
        print(f"{run:16} τ={tau:>9.5f}  EM {o['EM']:6.2f}  F1 {o['F1']:6.2f}  "
              f"từ chối {out['abstain_rate']:5.1f}%  broken {broken} -> {path.name}")


if __name__ == "__main__":
    main()
