"""Sinh ``report/generated/*.tex`` từ ``results/*.json`` — nguồn DUY NHẤT của số liệu báo cáo.

Báo cáo không gõ tay con số nào: văn bản dùng macro (``\\XlmrEM``), bảng được
sinh nguyên khối. Kết quả nào chưa có thì macro hiện chữ đỏ ``??`` và script in
cảnh báo — thiếu số phải NHÌN THẤY được, không được lặng lẽ thay bằng số đoán.

    python scripts/make_report_numbers.py          # rồi: cd report && latexmk -xelatex main.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrc.data import compute_stats, load_squad_file, split_by_context  # noqa: E402

sys.path.insert(0, str(_ROOT / "scripts"))
from _runs import PHOBERT_RUNS, best_dev_f1, primary_phobert  # noqa: E402

PHOBERT = primary_phobert(RES := _ROOT / "results")
RUN_LABEL = {"phobert": "PhoBERT, lr $3\\cdot10^{-5}$", "phobert_lr2e5": "PhoBERT, lr $2\\cdot10^{-5}$"}

OUT = _ROOT / "report" / "generated"
MISSING: list[str] = []

SYSTEMS = [  # (khoá tệp, tiền tố macro, tên hiển thị)
    ("abstain", "Abstain", "Luôn từ chối"),
    ("baseline", "Tfidf", "TF-IDF (truy hồi câu)"),
    ("xlmr", "Xlmr", "XLM-R-base"),
    (PHOBERT, "Phobert", "PhoBERT-base-v2"),
]
CAT_WORDS = {"E1": "EOne", "E2": "ETwo", "E3": "EThree", "E4": "EFour", "E5": "EFive"}
CAT_NAMES = {"E1": "Ranh giới từ ghép", "E2": "Ngữ cảnh gây nhiễu", "E3": "Câu hỏi phủ định",
             "E4": "Suy luận nhiều bước", "E5": "Nhãn mập mờ"}


def load(name: str):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def vi(x, digits: int = 2) -> str:
    """Định dạng số kiểu Việt Nam: 28.454 ; 30,44."""
    if x is None:
        return r"\missing{}"
    if isinstance(x, int) or (isinstance(x, float) and digits == 0):
        return f"{int(round(x)):,}".replace(",", ".")
    s = f"{x:,.{digits}f}"
    return s.replace(",", "X").replace(".", "{,}").replace("X", ".")


def pval(p) -> str:
    if p is None:
        return r"\missing{}"
    if p < 0.001:
        return r"$<$\,0{,}001"
    return vi(p, 3)


class Macros:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, name: str, value, digits: int = 2, raw: bool = False):
        if value is None:
            MISSING.append(name)
        text = value if raw and value is not None else vi(value, digits)
        self.lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")


def get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def table(path: str, body: str) -> None:
    (OUT / path).write_text(body, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = Macros()
    m.lines.append(r"\providecommand{\missing}{\textcolor{badred}{\textbf{??}}}")

    # ── dữ liệu ───────────────────────────────────────────────────────────────
    train = load_squad_file(_ROOT / "data/raw/viquad2_train.json")
    val = load_squad_file(_ROOT / "data/raw/viquad2_validation.json")
    fit, dev = split_by_context(train, val_frac=0.05, seed=42)
    for prefix, ex in (("Train", train), ("Val", val), ("Fit", fit), ("Dev", dev)):
        s = compute_stats(ex)
        m(f"N{prefix}", s["num_questions"])
        m(f"N{prefix}Ctx", s["num_contexts"])
        m(f"N{prefix}Imp", s["num_impossible"])
        m(f"Pct{prefix}Imp", s["impossible_pct"])
        m(f"N{prefix}Ans", s["num_with_gold"])

    # ── biên từ ───────────────────────────────────────────────────────────────
    seg = get(load("segmentation_validation.json"), "summary") or {}
    for k, name in (("n_located", "SegN"), ("n_aligned", "SegAligned"), ("n_misaligned", "SegMis"),
                    ("misaligned_start_cut", "SegStartCut"), ("misaligned_end_cut", "SegEndCut"),
                    ("n_contexts", "SegCtx")):
        m(name, seg.get(k), 0)
    for k, name in (("misaligned_pct", "SegMisPct"), ("oracle_em_all_pct", "SegOracle"),
                    ("oracle_em_misaligned_pct", "SegOracleMis")):
        m(name, seg.get(k))
    m("SegAlignedPct", 100 - seg["misaligned_pct"] if seg else None)
    m("SegCeilingLoss", 100 - seg["oracle_em_all_pct"] if seg else None)
    m("SegCtxCoverage",(seg.get("char_coverage_min") or 0) * 100 if seg else None)
    mis_labels = get(load("misaligned_labels.json"), "counts") or {}
    for k, name in (("segmenter_period", "MisPeriod"), ("segmenter_merge", "MisMerge"),
                    ("compound_boundary", "MisCompound"), ("annotation_truncated", "MisAnnot")):
        m(name, mis_labels.get(k), 0)

    # ── kiểm định stress-test ────────────────────────────────────────────────
    audit = load("stress_test_audit.json") or {}
    tot = audit.get("totals", {})
    for k, name in (("n", "AuditN"), ("answerable", "AuditAns"),
                    ("answerable_with_answer_in_context", "AuditAnsInCtx"),
                    ("answerable_with_correct_offset", "AuditOffsetOK"), ("valid", "AuditValid"),
                    ("distinct_contexts", "AuditCtx"), ("flat_records_outside_squad_schema", "AuditFlat")):
        m(name, tot.get(k), 0)
    m("AuditCombined", get(audit, "combined_file", "top_level_records"), 0)
    m("AuditTrainOverlap", get(audit, "overlap_with_viquad", "contexts_in_train"), 0)
    cats = audit.get("by_category", {})
    rows = []
    for code, c in cats.items():
        w = CAT_WORDS[code]
        m(f"Audit{w}Valid", c["valid"], 0)
        m(f"Audit{w}Ans", c["answerable"], 0)
        m(f"Audit{w}AnsInCtx", c["answer_in_context"], 0)
        m(f"Audit{w}DistinctQ", c["distinct_questions"], 0)
        rows.append(f"{code} & {CAT_NAMES[code]} & {c['n']} & {c['answerable']} & "
                    f"{c['answer_in_context']} & {c['offset_correct']} & {c['answer_in_question']} & "
                    f"{c['distinct_questions']} & \\textbf{{{c['valid']}}} \\\\")
    table("tab_audit.tex", "\n".join(rows) + "\n")

    # ── từng hệ thống ─────────────────────────────────────────────────────────
    diag = load("diagnosis_validation.json") or {}
    main_rows, stress_rows, tax_rows, len_rows, ctx_rows, qt_rows, align_rows = [], [], [], [], [], [], []
    for key, P, label in SYSTEMS:
        ev = load(f"eval_{key}_validation.json")
        st = load(f"eval_{key}_stress.json")
        dg = get(diag, "models", key)
        m(f"{P}EM", get(ev, "overall", "EM"))
        m(f"{P}Fone", get(ev, "overall", "F1"))
        m(f"{P}AnsEM", get(ev, "answerable_only", "EM"))
        m(f"{P}AnsFone", get(ev, "answerable_only", "F1"))
        m(f"{P}ImpEM", get(ev, "impossible_only", "EM"))
        m(f"{P}Lat", get(ev, "avg_latency_ms"))
        m(f"{P}StressEM", get(st, "overall", "EM"))
        m(f"{P}AbsRate", get(dg, "abstention", "abstain_rate"))
        m(f"{P}AbsPrec", get(dg, "abstention", "precision"))
        m(f"{P}AbsRec", get(dg, "abstention", "recall"))
        m(f"{P}AlEM", get(dg, "by_alignment", "aligned", "EM"))
        m(f"{P}MisEM", get(dg, "by_alignment", "misaligned", "EM"))
        m(f"{P}MisFone", get(dg, "by_alignment", "misaligned", "F1"))
        counts = get(dg, "taxonomy_all", "counts") or {}
        nerr = get(dg, "taxonomy_all", "n_errors")
        boundary = sum(counts.get(k, 0) for k in ("boundary_superset", "boundary_subset", "boundary_overlap")) if counts else None
        m(f"{P}NErr", nerr, 0)
        m(f"{P}ErrBoundary", boundary, 0)
        m(f"{P}ErrBoundaryPct", 100 * boundary / nerr if nerr else None)
        for k, w in (("false_abstain", "FalseAbs"), ("false_answer", "FalseAns"), ("wrong_span", "Wrong"),
                     ("boundary_superset", "Super"), ("boundary_subset", "Sub"), ("boundary_overlap", "Overlap")):
            m(f"{P}Err{w}", counts.get(k) if counts else None, 0)
            m(f"{P}Err{w}Pct", 100 * counts[k] / nerr if counts and nerr else None)

        if ev:
            a = get(dg, "abstention", "abstain_rate")
            main_rows.append(f"{label} & {vi(ev['overall']['EM'])} & {vi(ev['overall']['F1'])} & "
                             f"{vi(ev['answerable_only']['EM'])} & {vi(ev['answerable_only']['F1'])} & "
                             f"{vi(ev['impossible_only']['EM'])} & {vi(a)} & {vi(ev['avg_latency_ms'])} \\\\")
            for bucket, v in ev.get("by_context_length", {}).items():
                ctx_rows.append((bucket, label, v))
            for qt, v in ev.get("by_question_type", {}).items():
                if isinstance(v, dict):
                    qt_rows.append((qt, label, v))
        if st:
            cells = []
            for code in CAT_WORDS:
                c = get(st, "by_category", code) or {}
                allc, valc = c.get("all", {}), c.get("valid", {})
                cells.append(f"{vi(allc.get('EM'), 1)} / {vi(valc.get('EM'), 1)}")
            stress_rows.append(f"{label} & " + " & ".join(cells) + f" & {vi(st['overall']['EM'], 1)} \\\\")
        if counts:
            tax_rows.append(f"{label} & {nerr} & " + " & ".join(
                str(counts[k]) for k in ("false_abstain", "false_answer", "boundary_superset",
                                         "boundary_subset", "boundary_overlap", "wrong_span")) + " \\\\")
        if dg:
            al, mi = dg["by_alignment"]["aligned"], dg["by_alignment"]["misaligned"]
            align_rows.append(f"{label} & {vi(al['EM'])} & {vi(al['F1'])} & {vi(mi['EM'])} & {vi(mi['F1'])} \\\\")
            len_rows.append(f"{label} & " + " & ".join(
                f"{vi(v['EM'], 1)} / {vi(v['F1'], 1)}" for _, v in sorted(dg["by_answer_length"].items(),
                                                                         key=lambda kv: ["1-2", "3-5", "6-10", "11+"].index(kv[0]))) + " \\\\")

    table("tab_main.tex", "\n".join(main_rows) + "\n")
    table("tab_stress.tex", "\n".join(stress_rows) + "\n")
    table("tab_taxonomy.tex", "\n".join(tax_rows) + "\n")
    table("tab_alignment.tex", "\n".join(align_rows) + "\n")
    table("tab_answer_length.tex", "\n".join(len_rows) + "\n")
    slices = get(diag, "slices") or {}
    lens = slices.get("answer_length", {})
    for b, w in (("1-2", "One"), ("3-5", "Three"), ("6-10", "Six"), ("11+", "Eleven")):
        m(f"NLen{w}", lens.get(b), 0)

    def pivot(rows, order):
        labels = [lab for _, _, lab in SYSTEMS if any(r[1] == lab for r in rows)]
        out = []
        for bucket in order:
            vals = {r[1]: r[2] for r in rows if r[0] == bucket}
            if not vals:
                continue
            n = next(iter(vals.values()))["count"]
            flag = r"$^\dagger$" if n < 30 else ""
            out.append(f"{bucket}{flag} & {n} & " + " & ".join(
                f"{vi(vals[lab]['EM'], 1)} / {vi(vals[lab]['F1'], 1)}" if lab in vals else r"\missing{}"
                for lab in labels) + " \\\\")
        return "\n".join(out) + "\n"

    table("tab_context_length.tex", pivot(ctx_rows, ["<100", "100-200", "200-300", "300+"]))
    table("tab_qtype.tex", pivot(qt_rows, ["single-sentence", "multi-sentence"]))

    # ── so sánh cặp PhoBERT vs XLM-R ─────────────────────────────────────────
    pair = get(diag, "paired_phobert_vs_xlmr") or {}
    pair_rows = []
    for part, w in (("all", ""), ("answerable", "Ans"), ("impossible", "Imp"), ("misaligned", "Mis")):
        r = pair.get(part)
        m(f"Pair{w}N", get(r, "n"), 0)
        m(f"Pair{w}Diff", get(r, "em_diff"))
        m(f"Pair{w}Lo", (get(r, "ci95") or [None])[0])
        m(f"Pair{w}Hi", (get(r, "ci95") or [None, None])[1])
        m(f"Pair{w}P", pval(get(r, "mcnemar_p")) if r else None, raw=True)
        m(f"Pair{w}OnlyP", get(r, "only_a"), 0)
        m(f"Pair{w}OnlyX", get(r, "only_b"), 0)
        m(f"Pair{w}Both", get(r, "both_correct"), 0)
        m(f"Pair{w}Neither", get(r, "neither"), 0)
        if r:
            name = {"all": "Toàn bộ", "answerable": "Có đáp án", "impossible": "Không có đáp án",
                    "misaligned": "Lệch biên từ"}[part]
            pair_rows.append(f"{name} & {r['n']} & {r['both_correct']} & {r['only_a']} & {r['only_b']} & "
                             f"{r['neither']} & {vi(r['em_diff'])} & [{vi(r['ci95'][0])}; {vi(r['ci95'][1])}] & "
                             f"{pval(r['mcnemar_p'])} \\\\")
    table("tab_paired.tex", "\n".join(pair_rows) + "\n")

    # ── huấn luyện ────────────────────────────────────────────────────────────
    curve_rows = []
    for key, P, label in [("xlmr", "Xlmr", "XLM-R-base")] + [
            (r, "PhobertRun" + ("A" if r == "phobert" else "B"), RUN_LABEL[r]) for r in PHOBERT_RUNS]:
        c = load(f"training_curve_{key}.json")
        cfg = get(c, "config") or {}
        m(f"{P}BestEpoch", get(c, "best_epoch"), 0)
        m(f"{P}MaxLen", cfg.get("max_length"), 0)
        m(f"{P}Stride", cfg.get("doc_stride"), 0)
        epochs = get(c, "curve") or []
        m(f"{P}Epochs", len(epochs) if epochs else None, 0)
        m(f"{P}TrainMin", sum(e["seconds"] for e in epochs) / 60 if epochs else None, 0)
        for e in epochs:
            best = r"\,$\star$" if e["epoch"] == get(c, "best_epoch") else ""
            curve_rows.append(f"{label} & {e['epoch']}{best} & {vi(e['train_loss'], 4)} & "
                              f"{vi(e['val_em'])} & {vi(e['val_f1'])} & {vi(e['seconds'] / 60, 1)} \\\\")
    table("tab_curves.tex", "\n".join(curve_rows) + "\n")

    # ── hai lần chạy PhoBERT ─────────────────────────────────────────────────
    run_rows = []
    for r in PHOBERT_RUNS:
        ev, c = load(f"eval_{r}_validation.json"), load(f"training_curve_{r}.json")
        ab = get(diag, "models", r, "abstention", "abstain_rate")
        w = "A" if r == "phobert" else "B"
        m(f"PhobertRun{w}DevFone", best_dev_f1(RES, r))
        m(f"PhobertRun{w}EM", get(ev, "overall", "EM"))
        m(f"PhobertRun{w}AnsEM", get(ev, "answerable_only", "EM"))
        m(f"PhobertRun{w}ImpEM", get(ev, "impossible_only", "EM"))
        m(f"PhobertRun{w}AbsRate", ab)
        if ev and c:
            star = r" $\star$" if r == PHOBERT else ""
            run_rows.append(f"{RUN_LABEL[r]}{star} & {c['best_epoch']} & {vi(best_dev_f1(RES, r))} & "
                            f"{vi(ev['overall']['EM'])} & {vi(ev['overall']['F1'])} & "
                            f"{vi(ev['answerable_only']['EM'])} & {vi(ev['impossible_only']['EM'])} & {vi(ab)} \\\\")
    table("tab_phobert_runs.tex", "\n".join(run_rows) + "\n")
    m("PhobertPrimaryLabel", RUN_LABEL[PHOBERT], raw=True)

    (OUT / "numbers.tex").write_text(
        "% SINH TỰ ĐỘNG bởi scripts/make_report_numbers.py — KHÔNG SỬA TAY\n" + "\n".join(m.lines) + "\n",
        encoding="utf-8")
    print(f"{len(m.lines) - 1} macro -> {OUT / 'numbers.tex'}")
    if MISSING:
        print(f"CẢNH BÁO: {len(MISSING)} giá trị chưa có (hiện '??' trong PDF): "
              f"{', '.join(sorted(set(MISSING))[:12])}{' ...' if len(set(MISSING)) > 12 else ''}")


if __name__ == "__main__":
    main()
