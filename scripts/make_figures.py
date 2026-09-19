"""Hình cho báo cáo và slide, vẽ từ results/*.json -> report/figures/*.png.

Màu theo thực thể, cố định (không theo thứ hạng): XLM-R xanh dương, PhoBERT cam,
TF-IDF xanh ngọc; "luôn từ chối" là đường tham chiếu xám. Bảng palette đã qua
kiểm tra CVD; mọi hình đều có bảng số tương ứng trong báo cáo.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "report" / "figures"
PREFIX = ""  # "slide_" khi vẽ bản cho slide (chữ lớn hơn)
COLOR = {"xlmr": "#2a78d6", "phobert": "#eb6834", "baseline": "#1baf7a",
         "xlmr_seed13": "#8fb8e8", "phobert_seed13": "#f3a57e"}
LABEL = {"xlmr": "XLM-R (seed 42)", "phobert": "PhoBERT (seed 42)", "baseline": "TF-IDF", "abstain": "Luôn từ chối",
         "xlmr_seed13": "XLM-R (seed 13)", "phobert_seed13": "PhoBERT (seed 13)"}
# Hai seed được báo cáo ngang nhau.
BOTH_SEEDS = ["baseline", "xlmr", "xlmr_seed13", "phobert", "phobert_seed13"]
ERR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def models(keys):
    return [k for k in keys if load(f"eval_{k}_validation.json")]


def fig_main():
    ms = models(BOTH_SEEDS)
    groups = [("EM\ntoàn bộ", "overall", "EM"), ("F1\ntoàn bộ", "overall", "F1"),
              ("EM\ncâu có đáp án", "answerable_only", "EM"), ("F1\ncâu có đáp án", "answerable_only", "F1"),
              ("EM\ncâu không đáp án", "impossible_only", "EM")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / max(1, len(ms))
    for i, m in enumerate(ms):
        ev = load(f"eval_{m}_validation.json")
        vals = [ev[g][k] for _, g, k in groups]
        xs = [j + (i - (len(ms) - 1) / 2) * w for j in range(len(groups))]
        bars = ax.bar(xs, vals, w - 0.03, color=COLOR[m], label=LABEL[m])
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.5, color=INK)
    ab = load("eval_abstain_validation.json")
    if ab:
        ax.axhline(ab["overall"]["EM"], color=MUTED, lw=1.2, ls="--")
        ax.text(1.45, ab["overall"]["EM"] - 6.5, f"luôn từ chối: EM toàn bộ = {ab['overall']['EM']:.2f}".replace(".", ","),
                ha="left", fontsize=8, color=MUTED, zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID))
    ax.set_xticks(range(len(groups)), [g for g, _, _ in groups])
    ax.set_ylim(0, 105); ax.set_ylabel("Điểm (%)")
    ax.legend(ncol=len(ms), loc="upper left", bbox_to_anchor=(0, 1.13), fontsize=7.5)
    fig.savefig(OUT / (PREFIX + "main_results.png")); plt.close(fig)


def fig_taxonomy():
    d = load("diagnosis_validation.json")
    if not d:
        return
    ms = [m for m in BOTH_SEEDS if m in d["models"]]
    types = ["false_abstain", "false_answer", "boundary_superset", "boundary_subset", "boundary_overlap", "wrong_span"]
    names = ["Từ chối sai", "Trả lời sai", "Biên thừa", "Biên thiếu", "Biên lệch", "Sai vị trí"]
    fig, ax = plt.subplots(figsize=(9, 0.75 + 0.62 * len(ms)))
    for y, m in enumerate(ms):
        c = d["models"][m]["taxonomy_all"]["counts"]
        left = 0
        for t, col in zip(types, ERR_COLORS):
            ax.barh(y, c[t], left=left, color=col, edgecolor="white", linewidth=1.5, height=0.6)
            if c[t] >= 140:
                ax.text(left + c[t] / 2, y, str(c[t]), ha="center", va="center", fontsize=7.5, color="white")
            left += c[t]
        ax.text(left + 15, y, f"{left} lỗi", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(ms)), [LABEL[m] for m in ms]); ax.invert_yaxis()
    ax.set_xlabel(f"Số câu sai trên {d['n']} câu validation")
    ax.grid(axis="y", visible=False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in ERR_COLORS], names, ncol=6,
              loc="upper left", bbox_to_anchor=(0, 1.02 + 0.5 / len(ms)), fontsize=8)
    fig.savefig(OUT / (PREFIX + "error_taxonomy.png")); plt.close(fig)


def fig_lines(key, order, xlabel, fname, source):
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for m in ["baseline", "xlmr", "phobert"]:
        if source == "diag":
            d = load("diagnosis_validation.json")
            if not d or m not in d["models"]:
                continue
            data = d["models"][m][key]
        else:
            ev = load(f"eval_{m}_validation.json")
            if not ev:
                continue
            data = ev[key]
        xs = [b for b in order if b in data]
        ax.plot(range(len(xs)), [data[b]["EM"] for b in xs], color=COLOR[m], lw=2, marker="o", ms=6, label=LABEL[m])
        ax.set_xticks(range(len(xs)), [f"{b}\n(n={data[b]['count']})" for b in xs])
    ax.set_ylabel("EM (%)"); ax.set_xlabel(xlabel); ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    fig.savefig(OUT / (PREFIX + fname)); plt.close(fig)


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    runs = [("xlmr", "-", "o", "XLM-R, seed 42"),
            ("xlmr_seed13", "--", "s", "XLM-R, seed 13"),
            ("phobert", "-", "o", "PhoBERT 3e-5, seed 42"),
            ("phobert_seed13", "--", "s", "PhoBERT 3e-5, seed 13"),
            ("phobert_lr2e5", ":", "^", "PhoBERT 2e-5, seed 42"),
            ("phobert_stable", "-.", "D", "PhoBERT 1e-5, seed 42")]
    for m, ls, mk, lab in runs:
        c = load(f"training_curve_{m}.json")
        if not c:
            continue
        col = COLOR["xlmr" if m == "xlmr" else "phobert"]
        ep = [e["epoch"] for e in c["curve"]]
        axes[0].plot(ep, [e["train_loss"] for e in c["curve"]], color=col, lw=2, ls=ls, marker=mk, ms=6, label=lab)
        axes[1].plot(ep, [e["val_f1"] for e in c["curve"]], color=col, lw=2, ls=ls, marker=mk, ms=6, label=lab)
    axes[0].set_title("Loss huấn luyện (trung bình epoch)", color=INK)
    axes[1].set_title("F1 trên dev (tiêu chí chọn epoch)", color=INK)
    for a in axes:
        a.set_xticks([1, 2, 3]); a.set_xlabel("Epoch")
    axes[0].legend(fontsize=7)
    fig.savefig(OUT / (PREFIX + "training_curves.png")); plt.close(fig)


def fig_audit():
    a = load("stress_test_audit.json")
    if not a:
        return
    cats = list(a["by_category"])
    parts = [("Có đáp án, đáp án NẰM trong ngữ cảnh", "#2a78d6", lambda c: c["answer_in_context"]),
             ("Có đáp án, đáp án KHÔNG có trong ngữ cảnh", "#eb6834", lambda c: c["answerable"] - c["answer_in_context"]),
             ("Không có đáp án", "#b9b8b0", lambda c: c["unanswerable"])]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for j, cat in enumerate(cats):
        c, bottom = a["by_category"][cat], 0
        for name, col, f in parts:
            v = f(c)
            ax.bar(j, v, bottom=bottom, color=col, edgecolor="white", linewidth=1.5, width=0.6)
            if v >= 2:
                ax.text(j, bottom + v / 2, str(v), ha="center", va="center", fontsize=8,
                        color="white" if col != "#b9b8b0" else INK)
            bottom += v
    ax.set_xticks(range(len(cats)), cats); ax.set_ylabel("Số câu (mỗi nhóm 50)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in parts], [n for n, _, _ in parts],
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / (PREFIX + "stress_audit.png")); plt.close(fig)


def render():
    fig_main(); fig_taxonomy(); fig_curves(); fig_audit()
    fig_lines("by_answer_length", ["1-2", "3-5", "6-10", "11+"], "Độ dài đáp án vàng (âm tiết)", "by_answer_length.png", "diag")
    fig_lines("by_context_length", ["<100", "100-200", "200-300", "300+"], "Độ dài đoạn văn (âm tiết)", "by_context_length.png", "eval")


def main():
    global PREFIX
    OUT.mkdir(parents=True, exist_ok=True)
    render()
    # Bản cho slide: cùng dữ liệu, chữ lớn hơn để đọc được trên màn chiếu.
    PREFIX = "slide_"
    with plt.rc_context({"font.size": 13, "legend.fontsize": 11}):
        render()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
