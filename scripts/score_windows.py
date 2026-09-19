"""Lưu điểm TỪNG CỬA SỔ của một checkpoint trên dev hoặc validation.

Từ tệp này, dự đoán ở MỌI ngưỡng τ dựng lại được chính xác mà không chạy lại
model (``mrc.threshold.decide``). Dùng cho:

* chấm một checkpoint trên TOÀN BỘ dev (không phải mẫu 500 câu),
* chọn τ trên dev rồi áp một lần lên validation.

Dev = 5% context của train, tách bằng ``split_by_context(seed=42)`` — cùng tập với
``finetune.py --split-seed 42``.

    python scripts/score_windows.py --checkpoint models/phobert/epoch1 \
        --name phobert_epoch1 --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file, references_from, split_by_context  # noqa: E402
from mrc.metrics import evaluate  # noqa: E402
from mrc.threshold import predictions_at  # noqa: E402


def load_split(split: str, data_dir: str, split_seed: int, dev_frac: float):
    if split == "validation":
        return load_squad_file(Path(data_dir) / "viquad2_validation.json")
    train = load_squad_file(Path(data_dir) / "viquad2_train.json")
    _, dev = split_by_context(train, val_frac=dev_frac, seed=split_seed)
    return dev


def score_example(predictor, ex) -> list[dict]:
    from mrc.windowing import decode_span, make_windows, score_spans

    rows = []
    for window in make_windows(ex.question, ex.context, predictor.tokenizer,
                               max_length=predictor.max_length,
                               doc_stride=predictor.doc_stride):
        start_logits, end_logits = predictor._score_window(window)
        s = score_spans(start_logits, end_logits, window.offset_mapping,
                        max_answer_len=predictor.max_answer_len, top_k=1)
        if s is None or s.best is None:
            continue
        rows.append({"delta": s.null_delta, "score": s.best.score,
                     "answer": decode_span(ex.context, s.best.start_char, s.best.end_char)})
    return rows


def summary_at(records, refs, tau: float) -> dict:
    preds = predictions_at(records, tau)
    m = evaluate(preds, refs)
    n = len(preds)
    return {"tau": tau, "EM": round(m["EM"], 4), "F1": round(m["F1"], 4),
            "abstain_rate": round(100 * sum(1 for p in preds.values() if not p) / n, 4),
            "n": n}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True, help="tên tệp kết quả: windows_<name>_<split>.json")
    ap.add_argument("--split", choices=["dev", "validation"], required=True)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    args = ap.parse_args(argv)

    from mrc.transformer_qa import TransformerQA

    examples = load_split(args.split, args.data_dir, args.split_seed, args.dev_frac)
    predictor = TransformerQA.from_checkpoint(args.checkpoint, name=args.name)
    t0 = time.time()
    records = {ex.qid: score_example(predictor, ex) for ex in examples}
    refs = references_from(examples)

    import subprocess

    out = {
        "checkpoint": args.checkpoint, "split": args.split, "split_seed": args.split_seed,
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, cwd=_ROOT).stdout.strip(),
        "seconds": round(time.time() - t0, 1),
        "at_tau0": summary_at(records, refs, 0.0),
        "records": records,
    }
    path = Path("results") / f"windows_{args.name}_{args.split}.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{args.name} {args.split}: {out['at_tau0']}  ({out['seconds']}s) -> {path}")


if __name__ == "__main__":
    main()
