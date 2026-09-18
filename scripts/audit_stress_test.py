"""Kiểm định tự động bộ stress-test -> ``results/stress_test_audit.json``.

    python scripts/audit_stress_test.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import load_squad_file  # noqa: E402
from mrc.evaluate import _git_commit  # noqa: E402
from mrc.stress_test import CATEGORIES, load_stress_test  # noqa: E402


def main() -> None:
    examples, meta, file_stats = load_stress_test("data/stress_test")
    by_cat: dict[str, dict] = {}
    for code in CATEGORIES:
        items = [m for m in meta.values() if m["category"] == code]
        ans = [m for m in items if m["answerable"]]
        templates = Counter(m["template"] for m in items)
        by_cat[code] = {
            **file_stats[code],
            "n": len(items),
            "answerable": len(ans),
            "unanswerable": len(items) - len(ans),
            "answer_in_context": sum(bool(m["answer_in_context"]) for m in ans),
            "offset_correct": sum(bool(m["offset_correct"]) for m in ans),
            "answer_in_question": sum(bool(m["answer_in_question"]) for m in ans),
            "valid": sum(m["valid"] for m in items),
            "valid_answerable": sum(m["valid"] for m in ans),
            "distinct_questions": len({e.question for e in examples if meta[e.qid]["category"] == code}),
            "distinct_templates": len(templates),
            "top_template": templates.most_common(1)[0],
        }

    combined = json.loads(Path("data/stress_test/stress_test_combined.json").read_text())["data"]
    combined_qs = [q for a in combined if "paragraphs" in a for p in a["paragraphs"] for q in p["qas"]]
    train_ctx = {e.context for e in load_squad_file("data/raw/viquad2_train.json")}
    val_ctx = {e.context for e in load_squad_file("data/raw/viquad2_validation.json")}
    stress_ctx = {e.context for e in examples}

    total_ans = sum(c["answerable"] for c in by_cat.values())
    report = {
        "commit": _git_commit(),
        "by_category": by_cat,
        "totals": {
            "n": len(examples),
            "answerable": total_ans,
            "answerable_with_answer_in_context": sum(c["answer_in_context"] for c in by_cat.values()),
            "answerable_with_correct_offset": sum(c["offset_correct"] for c in by_cat.values()),
            "valid": sum(c["valid"] for c in by_cat.values()),
            "distinct_contexts": len(stress_ctx),
            "flat_records_outside_squad_schema": sum(c["flat_records"] for c in by_cat.values()),
        },
        "combined_file": {
            "top_level_records": len(combined),
            "squad_questions": len(combined_qs),
            "records_outside_squad_schema": sum("paragraphs" not in a for a in combined),
        },
        "overlap_with_viquad": {
            "contexts_in_train": len(stress_ctx & train_ctx),
            "contexts_in_validation": len(stress_ctx & val_ctx),
        },
    }
    out = Path("results/stress_test_audit.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("totals", "combined_file", "overlap_with_viquad")},
                     ensure_ascii=False, indent=2))
    for code, c in by_cat.items():
        print(code, {k: c[k] for k in ("n", "answerable", "answer_in_context", "offset_correct",
                                        "answer_in_question", "valid", "distinct_questions",
                                        "distinct_templates")})


if __name__ == "__main__":
    main()
