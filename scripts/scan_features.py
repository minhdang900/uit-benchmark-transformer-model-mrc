"""Kiểm tra toàn bộ feature huấn luyện có hợp lệ không (giả thuyết H2 của spec).

PhoBERT sụp đổ giữa epoch 2. Một giả thuyết là có feature hỏng — id ngoài từ
vựng, vị trí vượt bảng position embedding, nhãn "không có đáp án" không trỏ về
[CLS], hoặc nhãn span lệch khỏi đáp án — mà MPS nhận âm thầm thay vì báo lỗi.
Script này kiểm tra đúng những điều đó trên CHÍNH các feature finetune.py dùng.

    python scripts/scan_features.py --model vinai/phobert-base-v2 --word-segmented \
        --max-length 256 --doc-stride 96 --name phobert
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file, split_by_context  # noqa: E402
from mrc.features import prepare_train_features  # noqa: E402
from mrc.metrics import normalize_answer  # noqa: E402
from mrc.tokenization import load_tokenizer  # noqa: E402
from mrc.windowing import make_windows  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--word-segmented", action="store_true")
    ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--doc-stride", type=int, required=True)
    ap.add_argument("--data-dir", default="data/raw")
    args = ap.parse_args(argv)

    from transformers import AutoConfig

    tok = load_tokenizer(args.model, word_segmented=args.word_segmented)
    cfg = AutoConfig.from_pretrained(args.model)
    vocab = cfg.vocab_size
    pad_id = tok.pad_token_id
    cls_id = tok.cls_token_id
    unk_id = getattr(tok, "unk_token_id", None)
    # RoBERTa: position id = padding_idx + 1 + chỉ số token (token thật).
    pos_offset = (cfg.pad_token_id + 1) if cfg.model_type in ("roberta", "xlm-roberta") else 0

    train = load_squad_file(Path(args.data_dir) / "viquad2_train.json")
    train, _ = split_by_context(train, val_frac=0.05, seed=42)
    feats = prepare_train_features(train, tok, args.max_length, args.doc_stride)

    v = Counter()
    examples_bad: list[dict] = []
    unk = total_tokens = 0
    i = 0
    for ex_idx, ex in enumerate(train):
        for w in make_windows(ex.question, ex.context, tok,
                              max_length=args.max_length, doc_stride=args.doc_stride):
            ids = feats["input_ids"][i]
            assert feats["example_index"][i] == ex_idx, "thứ tự feature lệch"
            n_real = sum(feats["attention_mask"][i])
            s, e = feats["start_positions"][i], feats["end_positions"][i]
            i += 1

            real = ids[:n_real]
            total_tokens += n_real
            if unk_id is not None:
                unk += sum(1 for t in real if t == unk_id)
            if max(real) >= vocab or min(real) < 0:
                v["id_out_of_vocab"] += 1
            if n_real > args.max_length:
                v["too_long"] += 1
            if pos_offset + n_real - 1 >= cfg.max_position_embeddings:
                v["position_out_of_range"] += 1
            if any(t != pad_id for t in ids[n_real:]):
                v["mask_mismatch"] += 1
            cls_index = ids.index(cls_id)
            if cls_index != 0:
                v["cls_not_at_0"] += 1

            if ex.is_impossible:
                if (s, e) != (cls_index, cls_index):
                    v["impossible_not_cls"] += 1
                v["n_impossible"] += 1
                continue
            if (s, e) == (cls_index, cls_index):
                v["answerable_answer_outside_window"] += 1
                continue
            v["n_span"] += 1
            if not (0 < s <= e < n_real):
                v["span_index_invalid"] += 1
                continue
            offs = w.offset_mapping
            if offs[s] is None or offs[e] is None:
                v["span_on_non_context_token"] += 1
                continue
            text = ex.context[offs[s][0]:offs[e][1]]
            if normalize_answer(text) != normalize_answer(ex.answers[0]):
                v["span_text_mismatch"] += 1
                if len(examples_bad) < 25:
                    examples_bad.append({"qid": ex.qid, "gold": ex.answers[0], "labelled": text})

    assert i == len(feats["input_ids"]), "số feature không khớp"
    n = len(feats["input_ids"])
    out = {
        "model": args.model, "max_length": args.max_length, "doc_stride": args.doc_stride,
        "n_features": n, "vocab_size": vocab,
        "max_position_embeddings": cfg.max_position_embeddings,
        "unk_rate_pct": round(100 * unk / max(total_tokens, 1), 4),
        "counts": dict(v),
        "cls_label_fraction": round((v["n_impossible"] + v["answerable_answer_outside_window"]) / n, 4),
        "span_text_mismatch_pct": round(100 * v["span_text_mismatch"] / max(v["n_span"], 1), 4),
        "mismatch_examples": examples_bad,
    }
    path = Path("results") / f"feature_scan_{args.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "mismatch_examples"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
