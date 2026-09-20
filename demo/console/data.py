"""Nguồn số liệu cho ViMRC Console — đọc THẲNG từ ``results/*.json``.

Cùng nguyên tắc với báo cáo: không con số nào trong giao diện được gõ tay. Mỗi
hàm ở đây trả về đúng những gì một khối giao diện cần, đọc từ tệp kết quả đã
commit. Thiếu tệp thì trả ``None`` và khối đó tự hiện dấu — giống macro ``??``
của báo cáo: thiếu số phải NHÌN THẤY được, không lặng lẽ thay bằng số đoán.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# ── hệ thống hiển thị trong bảng chính (hai seed báo cáo NGANG NHAU) ──────────
SYSTEMS = [
    {"key": "abstain", "label": "Luôn từ chối", "short": "Luôn từ chối"},
    {"key": "baseline", "label": "TF-IDF (truy hồi câu)", "short": "TF-IDF"},
    {"key": "xlmr", "label": "XLM-R-base · seed 42", "short": "XLM-R s42"},
    {"key": "xlmr_seed13", "label": "XLM-R-base · seed 13", "short": "XLM-R s13"},
    {"key": "phobert", "label": "PhoBERT-base-v2 · seed 42", "short": "PhoBERT s42"},
    {"key": "phobert_seed13", "label": "PhoBERT-base-v2 · seed 13", "short": "PhoBERT s13"},
]

# ── các lần chạy huấn luyện (gồm cả lần chạy chẩn đoán) ──────────────────────
RUNS = [
    {"key": "xlmr", "name": "XLM-R-base · seed 42", "curve_label": "XLM-R s42", "color": "#56633f"},
    {"key": "xlmr_seed13", "name": "XLM-R-base · seed 13", "curve_label": "XLM-R s13", "color": "#8fa073"},
    {"key": "xlmr_256", "name": "XLM-R-base · 256/96 · seed 42", "curve_label": "XLM-R 256/96", "color": "#aebf92"},
    {"key": "phobert", "name": "PhoBERT · lr 3·10⁻⁵ · s42", "curve_label": "PhoBERT 3e-5 s42", "color": "#b2622d"},
    {"key": "phobert_seed13", "name": "PhoBERT · lr 3·10⁻⁵ · s13", "curve_label": "PhoBERT 3e-5 s13", "color": "#8c491a"},
    {"key": "phobert_lr2e5", "name": "PhoBERT · lr 2·10⁻⁵ · s42", "curve_label": "PhoBERT 2e-5 s42", "color": "#f6a06b"},
    {"key": "phobert_stable", "name": "PhoBERT · lr 10⁻⁵ · s42", "curve_label": "PhoBERT 1e-5 s42", "color": "#ffc6a5"},
]

CATEGORIES = [
    ("E1", "Ranh giới từ ghép", "ranh giới",
     "Đáp án nằm trong từ ghép đa âm tiết; dò xem lỗi tách từ có lan truyền vào mô hình không."),
    ("E2", "Ngữ cảnh gây nhiễu", "nhiễu",
     "Ngữ cảnh dài thêm và chèn thực thể nhiễu cùng loại; dò giới hạn độ dài và khả năng định vị."),
    ("E3", "Câu hỏi phủ định", "phủ định",
     "Câu hỏi chứa phủ định hoặc tiền giả định sai; dò xem mô hình hiểu phủ định hay chỉ học thiên lệch từ chối."),
    ("E4", "Suy luận nhiều bước", "suy luận",
     "Đáp án đòi hỏi nối hai mệnh đề cách xa nhau trong ngữ cảnh."),
    ("E5", "Nhãn mập mờ", "mập mờ",
     "Đáp án vàng có nhiều biến thể chấp nhận được; dò xem EM phạt oan tới mức nào."),
]

TAX_TYPES = [
    ("false_abstain", "từ chối sai", "#8c491a"),
    ("false_answer", "trả lời câu không có đáp án", "#d67f48"),
    ("boundary_superset", "biên thừa", "#f6a06b"),
    ("boundary_subset", "biên thiếu", "#ffc6a5"),
    ("boundary_overlap", "chồng lấn một phần", "#aebf92"),
    ("wrong_span", "sai hẳn", "#56633f"),
]


# ── tiện ích ─────────────────────────────────────────────────────────────────
def vi(x, digits: int = 2) -> str:
    """Số kiểu Việt Nam: 3.814 · 52,49. ``None`` -> dấu — (thiếu phải thấy được)."""
    if x is None:
        return "—"
    if isinstance(x, int) or digits == 0:
        return f"{int(round(x)):,}".replace(",", ".")
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@lru_cache(maxsize=None)
def load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def dig(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ── bảng kết quả chính trên validation ───────────────────────────────────────
MAIN_COLS = ["EM", "F1", "EM có đ.á.", "F1 có đ.á.", "EM không đ.á.", "Tỉ lệ từ chối", "ms/câu"]


def main_rows() -> list[dict]:
    diag = load("diagnosis_validation.json") or {}
    rows = []
    for s in SYSTEMS:
        ev = load(f"eval_{s['key']}_validation.json")
        if not ev:
            continue
        abstain = dig(diag, "models", s["key"], "abstention", "abstain_rate")
        rows.append({
            **s,
            "cells": [
                dig(ev, "overall", "EM"), dig(ev, "overall", "F1"),
                dig(ev, "answerable_only", "EM"), dig(ev, "answerable_only", "F1"),
                dig(ev, "impossible_only", "EM"), abstain, ev.get("avg_latency_ms"),
            ],
        })
    return rows


def tiles(system_key: str, baseline_key: str = "xlmr") -> list[dict]:
    """Bốn ô số lớn: EM, F1, tỉ lệ từ chối, độ trễ — kèm chênh lệch so với mốc."""
    rows = {r["key"]: r for r in main_rows()}
    cur, base = rows.get(system_key), rows.get(baseline_key)
    if not cur:
        return []
    ref = base or cur
    out = []
    for label, i, unit in (("EM validation", 0, "%"), ("F1 validation", 1, "%"),
                           ("Tỉ lệ từ chối", 5, "%"), ("Độ trễ / câu", 6, " ms")):
        v, b = cur["cells"][i], ref["cells"][i]
        if ref is cur or v is None or b is None:
            delta, good = "mốc so sánh", None
        else:
            d = v - b
            delta = f"{'+' if d >= 0 else '−'}{vi(abs(d))} so với {ref['short']}"
            good = d >= 0 if i != 6 else d <= 0
        out.append({"k": label, "v": (vi(v) + unit) if v is not None else "—",
                    "delta": delta, "good": good})
    return out


# ── ma trận stress-test E1–E5 ────────────────────────────────────────────────
def stress_matrix(scope: str = "all") -> tuple[list[dict], list[dict]]:
    """(cột, dòng). ``scope`` = ``all`` (250 câu) hoặc ``valid`` (145 câu hợp lệ)."""
    cols = [{"code": c, "name": n} for c, _, n, _ in CATEGORIES]
    rows = []
    total_valid = 0
    for s in SYSTEMS:
        ev = load(f"eval_{s['key']}_stress.json")
        if not ev:
            continue
        cells, num, den = [], 0.0, 0
        for code, _, _, _ in CATEGORIES:
            c = dig(ev, "by_category", code, scope) or {}
            cells.append(c.get("EM"))
            if c.get("EM") is not None and c.get("count"):
                num += c["EM"] * c["count"]
                den += c["count"]
        total_valid = den
        # Tổng của "toàn bộ" lấy từ overall đã chấm; của "hợp lệ" tính lại theo số câu.
        total = dig(ev, "overall", "EM") if scope == "all" else (num / den if den else None)
        rows.append({**s, "cells": cells + [total]})
    cols.append({"code": "Tổng", "name": f"{vi(250 if scope == 'all' else total_valid, 0)} câu"})
    return cols, rows


def stress_note(scope: str) -> str:
    audit = load("stress_test_audit.json") or {}
    n = dig(audit, "totals", "n")
    valid = dig(audit, "totals", "valid")
    ans = dig(audit, "totals", "answerable")
    abstain = dig(load("eval_abstain_stress.json") or {}, "overall", "EM")
    if scope == "all":
        return (f"Hệ thống “luôn từ chối” đạt {vi(abstain, 1)} EM trên toàn bộ {vi(n, 0)} câu — "
                f"cao hơn phần lớn mô hình thật. Đó là kết quả chẩn đoán, không phải xếp hạng: "
                f"bộ stress-test hiện tại thưởng cho việc từ chối, vì chỉ {vi(ans, 0)} câu có đáp án.")
    return (f"Lọc còn {vi(valid, 0)} câu qua kiểm định thì mọi hệ thống đều nhảy vọt và khoảng cách "
            f"giữa chúng thu hẹp lại — phần lớn chênh lệch ở bảng đầy đủ đến từ các câu hỏng, "
            f"không từ năng lực mô hình.")


# ── phân loại lỗi ────────────────────────────────────────────────────────────
def taxonomy_rows() -> list[dict]:
    diag = load("diagnosis_validation.json") or {}
    rows = []
    for s in SYSTEMS:
        counts = dig(diag, "models", s["key"], "taxonomy_all", "counts")
        if not counts:
            continue
        total = dig(diag, "models", s["key"], "taxonomy_all", "n_errors") or 0
        rows.append({
            **s, "total": total,
            "segs": [{"n": counts.get(k, 0), "label": label, "bg": bg,
                      "pct": (100 * counts.get(k, 0) / total) if total else 0}
                     for k, label, bg in TAX_TYPES],
        })
    return rows


# ── trang khám phá lỗi ───────────────────────────────────────────────────────
ERROR_FILTERS = [
    ("all", "Tất cả"),
    ("false_answer", "Trả lời câu không có đáp án"),
    ("false_abstain", "Từ chối sai"),
    ("boundary", "Lệch ranh giới"),
]


def _error_kinds(e: dict) -> set[str]:
    out = set()
    for side in ("phobert_error", "xlmr_error"):
        err = e.get(side)
        if not err:
            continue
        out.add("boundary" if str(err).startswith("boundary") else str(err))
    return out


def disagreements(kind: str = "all", limit: int = 24) -> tuple[list[dict], int]:
    """Các câu PhoBERT và XLM-R (seed 42) trả lời khác nhau, lọc theo cơ chế lỗi."""
    items = load("disagreements_phobert_xlmr.json") or []
    picked = items if kind == "all" else [e for e in items if kind in _error_kinds(e)]
    return picked[:limit], len(picked)


def paired_summary() -> str:
    d = load("diagnosis_validation.json") or {}
    p = dig(d, "paired_phobert_vs_xlmr", "all") or {}
    t = dig(d, "paired_phobert_vs_xlmr_tuned", "all") or {}
    ci = p.get("ci95_article") or [None, None]
    tci = t.get("ci95_article") or [None, None]
    return (f"{vi(p.get('only_a'), 0)} câu chỉ PhoBERT đúng và {vi(p.get('only_b'), 0)} câu chỉ XLM-R đúng "
            f"— {vi((p.get('only_a') or 0) + (p.get('only_b') or 0), 0)} câu bất đồng trên {vi(p.get('n'), 0)}. "
            f"Ở ngưỡng mặc định, chênh lệch EM tổng là {vi(p.get('em_diff'))} điểm "
            f"(KTC 95% theo bài viết {vi(ci[0])} … {vi(ci[1])}; p = {vi(p.get('mcnemar_p'), 3)}); "
            f"ở ngưỡng chọn trên dev là {vi(t.get('em_diff'))} điểm ({vi(tci[0])} … {vi(tci[1])}).")


# ── trang bộ stress-test ─────────────────────────────────────────────────────
def audit_tiles() -> list[dict]:
    a = load("stress_test_audit.json")
    if not a:
        return []
    t = a["totals"]
    pct = 100 * t["valid"] / t["n"] if t["n"] else 0
    return [
        {"k": "Cặp Q–A", "v": vi(t["n"], 0), "note": f"{len(CATEGORIES)} nhóm × 50"},
        {"k": "Qua kiểm định", "v": vi(t["valid"], 0),
         "note": f"{vi(pct, 0)}% — phần còn lại tính là không có đáp án"},
        {"k": "Có đáp án trong ngữ cảnh",
         "v": f"{vi(t['answerable_with_answer_in_context'], 0)}/{vi(t['answerable'], 0)}",
         "note": "lý do bộ này không dùng để xếp hạng"},
        {"k": "Trùng với train", "v": vi(dig(a, "overlap_with_viquad", "contexts_in_train"), 0),
         "note": "không rò rỉ dữ liệu"},
    ]


def group_cards() -> list[dict]:
    a = load("stress_test_audit.json") or {}
    out = []
    for code, name, _, mech in CATEGORIES:
        c = dig(a, "by_category", code) or {}
        out.append({"code": code, "name": name, "mech": mech, "n": c.get("n"),
                    "valid": c.get("valid"), "ans": c.get("answerable"),
                    "ans_in_ctx": c.get("answer_in_context"),
                    "distinct": c.get("distinct_questions"),
                    "top_template": (c.get("top_template") or [None, 0])})
    return out


def group_warning(code: str) -> str | None:
    """Cảnh báo dựng từ chính số kiểm định, không viết tay."""
    a = load("stress_test_audit.json") or {}
    c = dig(a, "by_category", code)
    if not c:
        return None
    bits = []
    if c["valid"] < c["n"]:
        bits.append(f"chỉ {c['valid']}/{c['n']} câu qua được kiểm định")
    if c["answerable"] and c["answer_in_context"] < c["answerable"]:
        bits.append(f"{c['answer_in_context']}/{c['answerable']} câu có đáp án thật sự "
                    f"chứa đáp án trong ngữ cảnh")
    if c["answerable"] == 0:
        bits.append("100% câu không có đáp án — hệ thống luôn từ chối đạt 100 EM "
                    "mà không chứng minh năng lực nào")
    if c["distinct_questions"] < c["n"]:
        bits.append(f"{c['distinct_questions']} câu hỏi khác nhau trên {c['n']} mục")
    tmpl = c.get("top_template") or [None, 0]
    if tmpl[0] and tmpl[1] >= 5:
        bits.append(f"khuôn mẫu lặp nhiều nhất xuất hiện {tmpl[1]} lần: “{tmpl[0]}”")
    if not bits:
        return None
    return "Nhóm này cần dựng lại trước khi rút kết luận: " + "; ".join(bits) + "."


def group_samples(code: str, limit: int = 6) -> list[dict]:
    """Mẫu thật từ bộ stress-test, kèm cờ hợp lệ theo kiểm định."""
    import sys

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    try:
        from mrc.stress_test import load_stress_test
    except Exception:
        return []
    try:
        examples, meta, _ = load_stress_test(str(ROOT / "data/stress_test"))
    except Exception:
        return []
    out = []
    for ex in examples:
        m = meta.get(ex.qid, {})
        if m.get("category") != code:
            continue
        out.append({"qid": ex.qid, "question": ex.question,
                    "gold": ex.answers[0] if ex.answers else "⌀ không có đáp án",
                    "valid": bool(m.get("valid"))})
        if len(out) >= limit:
            break
    return out


# ── trang huấn luyện & ngưỡng ────────────────────────────────────────────────
RUN_COLS = ["lr", "max len / stride", "epoch chọn", "F1 dev (đủ)", "nhánh CLS",
            "EM val", "tỉ lệ từ chối", "thời gian"]


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_text(lr) -> str:
    """3e-05 -> 3·10⁻⁵ (cột hẹp, dạng mũ dễ đọc và không bị ngắt dòng)."""
    if not lr:
        return "—"
    mant, exp = f"{lr:.0e}".split("e")
    return f"{mant}·10{str(int(exp)).translate(_SUP)}"


def collapse_from_probe(run_key: str) -> bool | None:
    """Với lần chạy không có nhật ký từng bước, đọc loss theo nhãn từ probe.

    ``loss_probe.json`` ghi loss nhánh "không có đáp án" của từng checkpoint. Tăng
    quá gấp đôi sau epoch 1 là sự kiện sụp đổ đã phân tích trong báo cáo.
    """
    ep = dig(load("loss_probe.json") or {}, "runs", run_key, "epochs")
    if not ep:
        return None
    vals = [ep[k]["cls"] for k in sorted(ep) if "cls" in ep[k]]
    if len(vals) < 2:
        return None
    return max(vals[1:]) > 2 * vals[0]


def run_rows() -> list[dict]:
    diag = load("diagnosis_validation.json") or {}
    thr = load("thresholds.json") or {}
    rows = []
    for r in RUNS:
        c = load(f"training_curve_{r['key']}.json")
        if not c:
            continue
        cfg = c.get("config") or {}
        ev = load(f"eval_{r['key']}_validation.json") or {}
        # Lần chạy trước khi có nhật ký từng bước KHÔNG có trường stability. Thiếu dữ
        # liệu phải hiện là thiếu — không được đọc thành "sụp đổ".
        has_stability = isinstance(c.get("stability"), dict)
        stable = dig(c, "stability", "stable")
        probe_collapse = None if has_stability else collapse_from_probe(r["key"])
        lr = cfg.get("lr")
        rows.append({
            **r,
            "curve": [e["val_f1"] for e in c.get("curve", [])],
            "cells": [
                lr_text(lr),
                f"{cfg.get('max_length')} / {cfg.get('doc_stride')}",
                str(c.get("best_epoch") or "—"),
                vi(dig(thr, "systems", r["key"], "dev_at_tau0", "F1")),
                ("ổn định" if stable else "sụp đổ") if has_stability
                else ("sụp đổ*" if probe_collapse
                      else ("ổn định*" if probe_collapse is False else "chưa đo")),
                vi(dig(ev, "overall", "EM")),
                vi(dig(diag, "models", r["key"], "abstention", "abstain_rate")),
                f"{vi(sum(e['seconds'] for e in c.get('curve', [])) / 60, 0)} ph",
            ],
            "stable": bool(stable) if has_stability else (probe_collapse is False),
            "has_stability": has_stability or probe_collapse is not None,
        })
    return rows


THR_COLS = ["τ chọn trên dev", "F1 tại τ=0", "F1 tại τ", "từ chối tại τ=0", "từ chối tại τ"]


def threshold_rows() -> list[dict]:
    thr = load("thresholds.json") or {}
    rows = []
    for r in RUNS:
        t = dig(thr, "systems", r["key"])
        if not t:
            continue
        rows.append({**r, "cells": [
            vi(t["tau"]), vi(dig(t, "validation_at_tau0", "F1")), vi(dig(t, "validation_at_tau", "F1")),
            vi(dig(t, "validation_at_tau0", "abstain_rate")), vi(dig(t, "validation_at_tau", "abstain_rate")),
        ]})
    return rows


def threshold_note() -> str:
    thr = load("thresholds.json") or {}
    a = dig(thr, "systems", "phobert") or {}
    return (f"Lần chạy sụp đổ nhánh CLS được τ cứu nhiều nhất: PhoBERT lr 3·10⁻⁵ seed 42 đi từ F1 "
            f"{vi(dig(a, 'validation_at_tau0', 'F1'))} lên {vi(dig(a, 'validation_at_tau', 'F1'))} "
            f"ở τ = {vi(a.get('tau'))} — ngang mức lần chạy seed 13 đạt được gần như không cần "
            f"chỉnh ngưỡng.")


# ── ranh giới từ ghép (trang hỏi đáp) ────────────────────────────────────────
SEG_CAUSE_TEXT = {
    "segmenter_merge": ("Nối nhầm âm tiết", "pyvi nối hai từ qua ranh giới thật."),
    "segmenter_period": ("Dính dấu chấm", "Luật viết tắt kéo dấu chấm vào token viết hoa."),
    "compound_boundary": ("Cắt giữa từ ghép", "Người gán nhãn chọn một phần của từ ghép."),
    "annotation_truncated": ("Nhãn bị cắt cụt", "Đáp án vàng đứt giữa âm tiết."),
}


def seg_summary() -> dict | None:
    s = dig(load("segmentation_validation.json") or {}, "summary")
    if not s:
        return None
    return {"aligned": s["n_aligned"], "answerable": s["n_answerable"],
            "aligned_pct": 100 - s["misaligned_pct"], "oracle": s["oracle_em_all_pct"],
            "ceiling_loss": 100 - s["oracle_em_all_pct"], "misaligned": s["n_misaligned"]}


def seg_causes() -> list[dict]:
    labels = load("misaligned_labels.json") or {}
    counts = dig(labels, "counts") or {}
    out = []
    for key, (name, desc) in SEG_CAUSE_TEXT.items():
        if key not in counts:
            continue
        out.append({"name": name, "n": counts[key], "desc": desc})
    return out


# ── provenance chân trang ────────────────────────────────────────────────────
def provenance() -> list[str]:
    # Lấy provenance từ lần chấm MỚI NHẤT có ghi phiên bản thư viện; các tệp cũ hơn
    # được chấm trước khi trường đó tồn tại.
    ev = {}
    for key in ("phobert_seed13", "xlmr_256", "phobert_stable", "xlmr_seed13", "phobert"):
        cand = load(f"eval_{key}_validation.json")
        if cand and (ev == {} or cand.get("library_versions")):
            ev = cand
        if ev.get("library_versions"):
            break
    lib = ev.get("library_versions") or {}
    commit = ev.get("commit") or "—"
    dev = dig(ev, "env", "device") or ev.get("device") or "—"
    return [
        f"UIT-ViQuAD 2.0 · validation {vi(dig(ev, 'overall', 'count'), 0)} câu · seed 42 / 13",
        f"commit {commit} · {dev} · torch {lib.get('torch') or '—'}",
        "CS221 · GVHD: TS. Đặng Văn Thìn",
    ]
