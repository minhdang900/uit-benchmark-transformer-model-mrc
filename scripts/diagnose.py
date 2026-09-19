"""Gộp dự đoán từng câu của mọi hệ thống -> ``results/diagnosis_validation.json``.

Cần có trước: ``results/predictions_<model>_validation.json`` (scripts/run_eval.py)
và ``results/segmentation_validation.json`` (scripts/analyze_segmentation.py).

    python scripts/diagnose.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file  # noqa: E402
from mrc.diagnosis import (  # noqa: E402
    abstention_stats, classify_error, error_taxonomy, paired_comparison, slice_scores,
)
from mrc.evaluate import _git_commit  # noqa: E402
from mrc.metrics import exact_match  # noqa: E402

sys.path.insert(0, str(_ROOT / "scripts"))
from _runs import primary_phobert  # noqa: E402

MODELS = ["abstain", "baseline", "xlmr", "phobert", "phobert_lr2e5", "phobert_stable",
          "xlmr_seed13", "phobert_seed13", "xlmr_256"]


def answer_length_bucket(n_syllables: int) -> str:
    if n_syllables <= 2:
        return "1-2"
    if n_syllables <= 5:
        return "3-5"
    if n_syllables <= 10:
        return "6-10"
    return "11+"


def main() -> None:
    examples = load_squad_file("data/raw/viquad2_validation.json")
    refs = {e.qid: list(e.answers) for e in examples}
    by_qid = {e.qid: e for e in examples}
    # Bài viết của mỗi câu — cho bootstrap theo BÀI (câu cùng bài không độc lập).
    article = {e.qid: e.title for e in examples}
    seg = json.loads(Path("results/segmentation_validation.json").read_text())["items"]

    preds = {}
    for m in MODELS:
        p = Path(f"results/predictions_{m}_validation.json")
        if p.exists():
            preds[m] = json.loads(p.read_text())
    print("models:", list(preds))

    answerable = [q for q, g in refs.items() if g]
    aligned = [q for q in answerable if seg.get(q, {}).get("aligned")]
    misaligned = [q for q in answerable if seg.get(q, {}).get("located") and not seg[q]["aligned"]]
    len_buckets: dict[str, list[str]] = {}
    for q in answerable:
        len_buckets.setdefault(answer_length_bucket(len(refs[q][0].split())), []).append(q)

    report: dict = {
        "commit": _git_commit(),
        "n": len(refs),
        "slices": {"answerable": len(answerable), "aligned": len(aligned),
                   "misaligned": len(misaligned),
                   "answer_length": {k: len(v) for k, v in sorted(len_buckets.items())}},
        "models": {},
    }
    # Đáp án "nghe hợp lý" (plausible_answers) mà người gán cố tình đặt vào câu
    # impossible của SQuAD 2.0 — KHÔNG phải nhãn, chỉ dùng để phân tích lỗi.
    raw = json.loads(Path("data/raw/viquad2_validation.json").read_text())["data"]
    plausible = {q["id"]: (q.get("plausible_answers") or {}).get("text", [])
                 for a in raw for para in a["paragraphs"] for q in para["qas"]}
    impossible = [q for q, g in refs.items() if not g]

    for m, p in preds.items():
        answered_imp = [q for q in impossible if p.get(q, "").strip()]
        report.setdefault("plausible_trap", {})[m] = {
            "answered_impossible": len(answered_imp),
            "equals_plausible_answer": sum(any(exact_match(p[q], t) for t in plausible.get(q, []))
                                           for q in answered_imp),
        }
        report["models"][m] = {
            "taxonomy_all": error_taxonomy(p, refs),
            "abstention": abstention_stats(p, refs),
            "by_alignment": {"aligned": slice_scores(p, refs, aligned),
                             "misaligned": slice_scores(p, refs, misaligned)},
            "by_answer_length": {k: slice_scores(p, refs, v) for k, v in sorted(len_buckets.items())},
        }

    em = {m: {q: max((exact_match(p.get(q, ""), g) for g in refs[q]), default=float(not p.get(q, "").strip()))
              for q in refs} for m, p in preds.items()}
    P = primary_phobert(Path("results"))
    report["phobert_primary"] = P
    print("PhoBERT chính (theo F1 dev):", P)
    if {"xlmr", P} <= set(preds):
        report["paired_phobert_vs_xlmr"] = {
            "all": paired_comparison(em[P], em["xlmr"], groups=article),
            "answerable": paired_comparison({q: em[P][q] for q in answerable},
                                            {q: em["xlmr"][q] for q in answerable},
                                            groups=article),
            "impossible": paired_comparison({q: em[P][q] for q in refs if not refs[q]},
                                            {q: em["xlmr"][q] for q in refs if not refs[q]},
                                            groups=article),
            "misaligned": paired_comparison({q: em[P][q] for q in misaligned},
                                            {q: em["xlmr"][q] for q in misaligned}),
        }
        # Cùng phép so sánh ở τ chọn trên dev (scripts/calibrate_thresholds.py): tách
        # "đọc giỏi" khỏi "sẵn lòng trả lời".
        tuned = {}
        for mdl in ("xlmr", P):
            tp = Path(f"results/predictions_{mdl}_tuned_validation.json")
            if tp.exists():
                t = json.loads(tp.read_text())
                tuned[mdl] = {q: max((exact_match(t.get(q, ""), g) for g in refs[q]),
                                     default=float(not t.get(q, "").strip())) for q in refs}
        if len(tuned) == 2:
            report["paired_phobert_vs_xlmr_tuned"] = {
                "all": paired_comparison(tuned[P], tuned["xlmr"], groups=article),
                "answerable": paired_comparison({q: tuned[P][q] for q in answerable},
                                                {q: tuned["xlmr"][q] for q in answerable},
                                                groups=article),
                "impossible": paired_comparison({q: tuned[P][q] for q in refs if not refs[q]},
                                                {q: tuned["xlmr"][q] for q in refs if not refs[q]},
                                                groups=article),
            }
        # Câu PhoBERT sai mà XLM-R đúng, và ngược lại: nguyên liệu cho phân tích định tính.
        disagree = []
        for q in refs:
            if em[P][q] != em["xlmr"][q]:
                disagree.append({
                    "qid": q, "question": by_qid[q].question, "gold": refs[q],
                    "phobert": preds[P].get(q, ""), "xlmr": preds["xlmr"].get(q, ""),
                    "winner": "phobert" if em[P][q] > em["xlmr"][q] else "xlmr",
                    "phobert_error": classify_error(preds[P].get(q, ""), refs[q]),
                    "xlmr_error": classify_error(preds["xlmr"].get(q, ""), refs[q]),
                    "aligned": seg.get(q, {}).get("aligned"),
                })
        Path("results/disagreements_phobert_xlmr.json").write_text(
            json.dumps(disagree, ensure_ascii=False, indent=1), encoding="utf-8")

    # Mẫu 100 lỗi ngẫu nhiên / model (seed 42) cho phần phân tích định tính.
    for m in ("xlmr", P):
        if m not in preds:
            continue
        errs = [q for q in refs if em[m][q] < 1]
        sample = random.Random(42).sample(errs, min(100, len(errs)))
        Path(f"results/error_sample_{m}.json").write_text(json.dumps([
            {"qid": q, "question": by_qid[q].question, "gold": refs[q],
             "prediction": preds[m].get(q, ""), "type": classify_error(preds[m].get(q, ""), refs[q]),
             "context": by_qid[q].context}
            for q in sample], ensure_ascii=False, indent=1), encoding="utf-8")

    out = Path("results/diagnosis_validation.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for m, r in report["models"].items():
        print(m, "abstain", r["abstention"]["abstain_rate"], "taxonomy", r["taxonomy_all"]["counts"])
        print("   aligned", r["by_alignment"]["aligned"], "misaligned", r["by_alignment"]["misaligned"])
    if "paired_phobert_vs_xlmr" in report:
        print(json.dumps(report["paired_phobert_vs_xlmr"], indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
