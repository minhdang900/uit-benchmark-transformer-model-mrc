"""Chấm một hoặc nhiều hệ thống, ghi kết quả có provenance + dự đoán từng câu.

    python scripts/run_eval.py --models abstain baseline xlmr phobert            # ViQuAD validation (toàn bộ)
    python scripts/run_eval.py --models abstain baseline xlmr phobert --dataset stress2   # bộ stress-test v2

Validation chính thức đóng vai tập TEST của đồ án: không model nào nhìn thấy nó
khi huấn luyện hay chọn epoch (xem scripts/finetune.py). Mỗi hệ thống được chấm
đúng một lần trên toàn bộ 3.814 câu — không lấy mẫu con.

Ghi ra:
    results/eval_<model>_<dataset>.json          số tổng + breakdown + provenance
    results/predictions_<model>_<dataset>.json   {qid: dự đoán} cho phân tích lỗi
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

from mrc.data import assert_gradeable, compute_stats, load_squad_file  # noqa: E402
from mrc.evaluate import breakdown, run_evaluation  # noqa: E402

NAMES = {
    "abstain": "Always-abstain",
    "baseline": "TF-IDF sentence retrieval",
    "xlmr": "XLM-R-base (fine-tuned)",
    "phobert": "PhoBERT-base-v2 (fine-tuned)",
    # Lần chạy lại với lr 2e-5 sau khi lần đầu (lr 3e-5) phân kỳ ở epoch 2.
    "phobert_lr2e5": "PhoBERT-base-v2 (fine-tuned, lr 2e-5)",
    # Lần chạy ổn định sau chẩn đoán (results/probe_resume.json) và các seed/cấu hình phụ.
    "phobert_stable": "PhoBERT-base-v2 (fine-tuned, lr 1e-5, stable)",
    "phobert_seed13": "PhoBERT-base-v2 (fine-tuned, seed 13)",
    "xlmr_seed13": "XLM-R-base (fine-tuned, seed 13)",
    "xlmr_256": "XLM-R-base (fine-tuned, 256/96)",
}


STRESS_V2_PATH = Path("data/stress_test_v2/stress_v2.json")


def build_predictor(kind: str):
    if kind == "abstain":
        from mrc.baselines import AlwaysAbstain

        return AlwaysAbstain()
    if kind == "baseline":
        from mrc.baseline_tfidf import TfidfRetriever

        return TfidfRetriever()

    from mrc.transformer_qa import TransformerQA

    path = Path("models") / kind
    if not (path / "config.json").exists():
        raise SystemExit(f"Chưa có model tại {path}. Chạy scripts/finetune.py trước.")
    # Cấu hình tokenize (word_segmented, max_length...) đọc từ checkpoint.
    return TransformerQA.from_checkpoint(str(path), name=NAMES[kind])


def load(dataset: str, data_dir: str):
    if dataset == "stress2":
        from mrc.stress_v2 import load_stress_v2

        return load_stress_v2(STRESS_V2_PATH)
    return load_squad_file(Path(data_dir) / f"viquad2_{dataset}.json"), None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(NAMES))
    ap.add_argument("--dataset", default="validation", choices=["validation", "stress2"],
                    help="validation | stress2 (bộ stress-test v1 đã gỡ bỏ)")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)

    examples, meta = load(args.dataset, args.data_dir)
    assert_gradeable(examples)
    stats = compute_stats(examples)
    print(f"dataset={args.dataset}  n={stats['num_questions']}  "
          f"impossible={stats['num_impossible']} ({stats['impossible_pct']}%)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in args.models:
        print(f"\n=== {kind} ===", flush=True)
        predictor = build_predictor(kind)
        result = run_evaluation(predictor, examples, split=args.dataset,
                                dataset="stress-test v2" if args.dataset == "stress2"
                                else "UIT-ViQuAD 2.0")
        predictions = result.pop("_predictions")
        result["config"] = {k: getattr(predictor, k) for k in
                            ("max_length", "doc_stride", "max_answer_len", "word_segmented")
                            if hasattr(predictor, k)}
        # Provenance huấn luyện: seed, lr, cửa sổ, thiết bị, phiên bản thư viện.
        curve = Path("models") / kind / "training_curve.json"
        if curve.exists():
            tc = json.loads(curve.read_text())
            result["seed"] = (tc.get("config") or {}).get("seed")
            result["training"] = {"config": tc.get("config"), "best_epoch": tc.get("best_epoch"),
                                  "stability": tc.get("stability"),
                                  "provenance": tc.get("provenance")}
        import torch
        import transformers

        result["library_versions"] = {"torch": torch.__version__,
                                      "transformers": transformers.__version__}
        if args.dataset == "stress2":
            from mrc.stress_v2 import score_report

            result.update(score_report(predictions, examples, meta))

        o, a, i = result["overall"], result["answerable_only"], result["impossible_only"]
        print(f"  overall    EM {o['EM']:6.2f}  F1 {o['F1']:6.2f}  (n={o['count']})")
        print(f"  answerable EM {a['EM']:6.2f}  F1 {a['F1']:6.2f}  (n={a['count']})")
        print(f"  impossible EM {i['EM']:6.2f}  (n={i['count']})   {result['avg_latency_ms']} ms/câu")

        (out_dir / f"eval_{kind}_{args.dataset}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"predictions_{kind}_{args.dataset}.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
