"""Tách loss huấn luyện theo loại nhãn, cho từng checkpoint epoch.

Câu hỏi: khi loss huấn luyện của PhoBERT TĂNG trong epoch 2, phần tăng nằm ở đâu?
Mỗi feature huấn luyện có một trong hai loại nhãn:

* ``cls``  — nhãn trỏ về token đầu (câu impossible, HOẶC cửa sổ không chứa đáp án);
* ``span`` — nhãn là vị trí đáp án thật.

Script lấy cùng một mẫu feature cố định từ tập fit, tính loss của từng checkpoint
``epoch1..3`` trên hai nhóm, và đếm tỉ lệ feature ``cls`` của mỗi tokenizer.

    python scripts/probe_loss.py --runs xlmr phobert phobert_lr2e5 --n 3000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file, split_by_context  # noqa: E402
from mrc.device import pick_device  # noqa: E402
from mrc.evaluate import _git_commit  # noqa: E402
from mrc.features import prepare_train_features  # noqa: E402
from mrc.tokenization import load_tokenizer, read_tokenization_config  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["xlmr", "phobert", "phobert_lr2e5"])
    ap.add_argument("--n", type=int, default=3000, help="số câu hỏi fit lấy mẫu")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForQuestionAnswering

    fit, _dev = split_by_context(load_squad_file("data/raw/viquad2_train.json"), 0.05, seed=42)
    sample = random.Random(0).sample(fit, args.n)
    device = pick_device()
    out: dict = {"commit": _git_commit(), "n_questions": args.n, "runs": {}}

    for run in args.runs:
        base = Path("models") / run
        cfg = read_tokenization_config(base / "epoch1")
        tok = load_tokenizer(str(base / "epoch1"), word_segmented=cfg.get("word_segmented", False))
        feats = prepare_train_features(sample, tok, cfg.get("max_length", 384), cfg.get("doc_stride", 128))
        n = len(feats["input_ids"])
        # Token CLS/<s> luôn ở vị trí 0 với cả hai mô hình (định dạng RoBERTa).
        is_cls = [s == 0 for s in feats["start_positions"]]
        rec = {"n_features": n, "cls_fraction": round(sum(is_cls) / n, 4),
               "impossible_fraction_questions": round(sum(e.is_impossible for e in sample) / args.n, 4),
               "epochs": {}}
        for ep in (1, 2, 3):
            ck = base / f"epoch{ep}"
            if not ck.exists():
                continue
            model = AutoModelForQuestionAnswering.from_pretrained(ck).to(device).eval()
            sums = {"cls": [0.0, 0], "span": [0.0, 0]}
            for i in range(0, n, args.batch):
                sl = slice(i, i + args.batch)
                batch = {k: torch.tensor(feats[k][sl], device=device)
                         for k in ("input_ids", "attention_mask", "start_positions", "end_positions")}
                with torch.no_grad():
                    o = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                ce = torch.nn.functional.cross_entropy
                per = (ce(o.start_logits, batch["start_positions"], reduction="none")
                       + ce(o.end_logits, batch["end_positions"], reduction="none")) / 2
                for j, v in enumerate(per.float().cpu().tolist()):
                    key = "cls" if is_cls[i + j] else "span"
                    sums[key][0] += v
                    sums[key][1] += 1
            rec["epochs"][ep] = {k: round(s / c, 4) if c else None for k, (s, c) in sums.items()}
            print(run, ep, rec["epochs"][ep], flush=True)
            del model
        out["runs"][run] = rec
        print(run, "features", n, "cls_fraction", rec["cls_fraction"], flush=True)

    Path("results/loss_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
