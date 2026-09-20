"""Dựng bộ stress-test v2 từ UIT-ViQuAD 2.0 validation và kiểm định nó.

    python scripts/build_stress_v2.py

Ghi ra:
    data/stress_test_v2/stress_v2.json      bộ dữ liệu (SQuAD-2.0 + khoá ``stress``)
    data/stress_test_v2/review_sheet.csv    mọi câu PERTURBATION có rủi ro nhãn (E2c, E3b)
                                            để hai người kiểm tra tay
    results/stress_v2_audit.json            thống kê + kiểm định tự động

Kiểm tra tay: điền cột ``nguoi_1`` / ``nguoi_2`` bằng ``ok`` hoặc ``loai``. Chạy lại
script: câu nào có ít nhất một người ghi ``loai`` bị loại (cùng câu gốc đi cặp),
và tỉ lệ đồng thuận giữa hai người được ghi vào audit. Script đọc quyết định cũ
trước khi ghi lại tệp, nên chạy lại không làm mất phần đã điền.

Script dừng với mã lỗi ≠ 0 nếu kiểm định tự động phát hiện vi phạm.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file  # noqa: E402
from mrc.evaluate import _git_commit  # noqa: E402
from mrc.stress_v2 import audit, build, to_squad  # noqa: E402

REVIEW_SUBSETS = ("E2c", "E3b")
REVIEW_FIELDS = ["qid", "subset", "cau_hoi", "dap_an_goc", "thay_doi", "context_moi",
                 "tieu_chi", "nguoi_1", "nguoi_2", "ghi_chu"]
CRITERIA = {
    "E2c": "Câu nhiễu KHÔNG trả lời được câu hỏi, và đáp án vàng vẫn đúng duy nhất?",
    "E3b": "Sau khi xoá câu, context KHÔNG còn chứa (kể cả diễn đạt khác) câu trả lời?",
}


def read_decisions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["qid"]: r for r in csv.DictReader(f)}


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """κ của Cohen cho hai người, nhãn ``ok``/``loai``."""
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    labels = {"ok", "loai"}
    pe = sum((sum(a == lab for a, _ in pairs) / n) * (sum(b == lab for _, b in pairs) / n)
             for lab in labels)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/stress_test_v2")
    ap.add_argument("--audit", default="results/stress_v2_audit.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    validation = load_squad_file(Path(args.data_dir) / "viquad2_validation.json")
    train_contexts = {e.context for e in load_squad_file(Path(args.data_dir) / "viquad2_train.json")}
    items = build(validation, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_path = out_dir / "review_sheet.csv"
    decisions = read_decisions(review_path)

    # Loại câu mà ít nhất một người đánh "loai" — kèm câu gốc đi cặp.
    rejected = {q for q, r in decisions.items()
                if "loai" in (r.get("nguoi_1", "").strip().lower(), r.get("nguoi_2", "").strip().lower())}
    kept = [it for it in items if it.qid not in rejected and it.qid.removesuffix("-orig") not in rejected]

    report = audit(kept, train_contexts)
    reviewed = [(r["nguoi_1"].strip().lower(), r["nguoi_2"].strip().lower()) for r in decisions.values()
                if r.get("nguoi_1", "").strip() and r.get("nguoi_2", "").strip()]
    report["manual_review"] = {
        "subsets": list(REVIEW_SUBSETS),
        "n_to_review": sum(it.subset in REVIEW_SUBSETS and it.role == "perturbed" for it in items),
        "n_reviewed_by_both": len(reviewed),
        "n_rejected": len(rejected),
        "agreement_pct": round(100 * sum(a == b for a, b in reviewed) / len(reviewed), 2)
        if reviewed else None,
        "cohen_kappa": cohen_kappa(reviewed),
        "status": "chưa kiểm tra tay" if not reviewed else "đã kiểm tra một phần hoặc toàn bộ",
    }
    report["commit"] = _git_commit()
    report["seed"] = args.seed
    report["source"] = "UIT-ViQuAD 2.0 validation"

    (out_dir / "stress_v2.json").write_text(
        json.dumps(to_squad(kept), ensure_ascii=False, indent=1), encoding="utf-8")

    with review_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        by_id = {it.qid: it for it in items}
        for it in items:
            if it.subset not in REVIEW_SUBSETS or it.role != "perturbed":
                continue
            twin = by_id[f"{it.qid}-orig"]
            old = decisions.get(it.qid, {})
            w.writerow({
                "qid": it.qid, "subset": it.subset, "cau_hoi": it.question,
                "dap_an_goc": " | ".join(twin.answers), "thay_doi": it.note,
                "context_moi": it.context, "tieu_chi": CRITERIA[it.subset],
                "nguoi_1": old.get("nguoi_1", ""), "nguoi_2": old.get("nguoi_2", ""),
                "ghi_chu": old.get("ghi_chu", ""),
            })

    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{report['n_items']} câu ({report['n_answerable']} có đáp án, "
          f"{report['n_impossible']} không), {report['distinct_contexts']} context")
    for code, c in report["by_category"].items():
        print(f"  {code} {c['name']:<28} n={c['n_items']:4d} (+{c['n_original_twins']} gốc)  luôn-từ-chối EM={c['always_abstain_em']}")
    print(f"vi phạm: {report['n_violations']}   loại sau kiểm tra tay: {len(rejected)}")
    if report["n_violations"]:
        for v in report["violations"][:10]:
            print("  ", v)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
