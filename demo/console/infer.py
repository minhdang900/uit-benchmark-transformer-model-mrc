"""Chạy mô hình thật cho trang Hỏi đáp.

Dùng đúng lớp inference của báo cáo (:class:`mrc.transformer_qa.TransformerQA`) và
đúng ngưỡng τ đã chọn trên dev (``results/thresholds.json``), nên đáp án hiện trên
giao diện là đáp án mà bảng kết quả đã chấm — không phải một đường chạy riêng.

Checkpoint nằm trong ``models/<run>/`` và KHÔNG được commit (xem .gitignore). Thiếu
checkpoint thì hàm báo lỗi rõ ràng để giao diện nói cần huấn luyện trước, thay vì
hiện một đáp án bịa.
"""
from __future__ import annotations

import sys

from .data import ROOT, dig, load

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Các mô hình chạy trực tiếp được trên trang Hỏi đáp.
DEMO_RUNS = [
    {"key": "phobert", "label": "PhoBERT s42", "dir": "phobert"},
    {"key": "phobert_seed13", "label": "PhoBERT s13", "dir": "phobert_seed13"},
    {"key": "xlmr", "label": "XLM-R s42", "dir": "xlmr"},
    {"key": "xlmr_seed13", "label": "XLM-R s13", "dir": "xlmr_seed13"},
]


def available_runs() -> list[dict]:
    """Chỉ những mô hình đã có checkpoint trên máy này."""
    return [r for r in DEMO_RUNS if (ROOT / "models" / r["dir"] / "config.json").exists()]


def tau_for(run_key: str) -> float:
    """τ chọn trên dev cho lần chạy này; không có thì 0."""
    t = dig(load("thresholds.json") or {}, "systems", run_key, "tau")
    return float(t) if isinstance(t, (int, float)) else 0.0


def load_predictor(run_key: str):
    """Nạp checkpoint (được Streamlit cache ở tầng gọi)."""
    from mrc.transformer_qa import TransformerQA

    run = next((r for r in DEMO_RUNS if r["key"] == run_key), None)
    if run is None:
        raise KeyError(f"Không biết lần chạy {run_key!r}")
    path = ROOT / "models" / run["dir"]
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"Chưa có checkpoint tại {path.relative_to(ROOT)} — chạy scripts/finetune.py trước.")
    return TransformerQA.from_checkpoint(str(path), name=run["label"])


def predict(predictor, context: str, question: str, tau: float) -> dict:
    """Dự đoán kèm bằng chứng: đáp án, biên độ so với null, span ký tự."""
    predictor.null_threshold = tau
    detail = predictor.predict_detailed(context, question)
    return {"answer": detail["answer"], "found": detail["found"],
            "span": detail["span"], "null_delta": detail["null_delta"]}


def score(prediction: str, golds: list[str]) -> tuple[float, float]:
    """(EM, F1) của một dự đoán so với đáp án vàng — dùng hàm chấm của báo cáo."""
    from mrc.metrics import exact_match, metric_max_over_ground_truths, token_f1

    if not golds:  # câu không có đáp án: đúng khi và chỉ khi trả rỗng
        ok = float(not prediction.strip())
        return ok, ok
    return (metric_max_over_ground_truths(exact_match, prediction, golds),
            metric_max_over_ground_truths(token_f1, prediction, golds))


# ── câu mẫu: lấy THẬT từ validation, chọn theo cơ chế lỗi ────────────────────
# Nhãn ngắn: chip nằm trong cột hẹp, nhãn dài sẽ bị cắt giữa chữ.
_WANTED = [
    ("false_answer", "trả lời sai"),
    ("false_abstain", "từ chối sai"),
    ("boundary_superset", "biên thừa"),
    ("boundary_subset", "biên thiếu"),
]


def examples(limit_per_kind: int = 1) -> list[dict]:
    """Các câu PhoBERT/XLM-R bất đồng, mỗi cơ chế lỗi một câu, kèm ngữ cảnh thật."""
    from mrc.data import load_squad_file

    val = load_squad_file(ROOT / "data/raw/viquad2_validation.json")
    by_qid = {e.qid: e for e in val}
    disagree = load("disagreements_phobert_xlmr.json") or []

    out: list[dict] = []
    seen: dict[str, int] = {}
    for e in disagree:
        ex = by_qid.get(e["qid"])
        if ex is None:
            continue
        for kind, label in _WANTED:
            if seen.get(kind, 0) >= limit_per_kind:
                continue
            if e.get("phobert_error") == kind or e.get("xlmr_error") == kind:
                seen[kind] = seen.get(kind, 0) + 1
                out.append({
                    "qid": ex.qid, "label": f"{ex.qid} · {label}",
                    "context": ex.context, "question": ex.question,
                    "gold": list(ex.answers),
                })
                break
    return out
