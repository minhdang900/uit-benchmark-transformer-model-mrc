"""Phân tích khám phá (EDA) UIT-ViQuAD 2.0: dữ liệu được xây dựng thế nào và điều đó
ảnh hưởng tới mô hình ra sao.

Chỉ dùng thư viện chuẩn + mrc.metrics / mrc.normalize (cùng quy ước chấm điểm).
Đầu ra: results/dataset_analysis.json

    python scripts/analyze_dataset.py
"""
from __future__ import annotations

import json
import re
import statistics as st
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mrc.metrics import exact_match, metric_max_over_ground_truths, token_f1  # noqa: E402
from mrc.normalize import tokenize  # noqa: E402

RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
MODELS = {"phobert_seed13": "PhoBERT s13", "xlmr_seed13": "XLM-R s13"}

QTYPES = [  # thứ tự = ưu tiên khi một câu chứa nhiều từ hỏi
    ("bao nhiêu", "Số lượng"), ("mấy", "Số lượng"),
    ("khi nào", "Thời gian"), ("năm nào", "Thời gian"), ("năm bao", "Thời gian"),
    ("lúc nào", "Thời gian"), ("thời gian nào", "Thời gian"), ("bao giờ", "Thời gian"),
    ("ở đâu", "Địa điểm"), ("nơi nào", "Địa điểm"), ("đâu", "Địa điểm"),
    ("tại sao", "Lý do"), ("vì sao", "Lý do"), ("lý do", "Lý do"), ("nguyên nhân", "Lý do"),
    ("như thế nào", "Cách thức"), ("ra sao", "Cách thức"), ("bằng cách nào", "Cách thức"),
    (" ai", "Người"), ("ai ", "Người"),
    ("là gì", "Định nghĩa/Sự vật"), ("gì", "Định nghĩa/Sự vật"),
    ("nào", "Lựa chọn (…nào)"),
]


def qtype(q: str) -> str:
    w = " " + q.lower().strip(" ?") + " "
    for k, lab in QTYPES:
        if k in w:
            return lab
    return "Khác"


def atype(a: str) -> str:
    toks = a.split()
    if re.search(r"\d", a):
        return "Số / ngày tháng"
    if toks and all(t[:1].isupper() for t in toks if t[:1].isalpha()) and len(toks) <= 6:
        return "Tên riêng"
    if len(toks) <= 4:
        return "Cụm ngắn (≤4 từ)"
    if len(toks) <= 12:
        return "Cụm vừa (5–12 từ)"
    return "Mệnh đề dài (>12 từ)"


SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"“(]?[A-ZÀ-Ỹ0-9])")


def sentences(ctx: str):
    out, pos = [], 0
    for part in SENT_RE.split(ctx):
        start = ctx.find(part, pos)
        out.append((start, start + len(part), part))
        pos = start + len(part)
    return out


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


ANTONYMS = [("trước", "sau"), ("nhiều", "ít"), ("lớn", "nhỏ"), ("cao", "thấp"), ("đầu", "cuối"),
            ("tăng", "giảm"), ("mạnh", "yếu"), ("thắng", "thua"), ("bắc", "nam"), ("đông", "tây"),
            ("trong", "ngoài"), ("dài", "ngắn"), ("mới", "cũ"), ("nhất", "kém")]
NEG = {"không", "chưa", "chẳng", "đừng", "chả"}


def edit_type(imp_q: str, sib_q: str) -> str:
    """Phân loại (heuristic) cách người gán nhãn biến câu có đáp án thành câu không đáp án."""
    a, b = tokenize(sib_q), tokenize(imp_q)
    ca, cb = Counter(a), Counter(b)
    add, rem = cb - ca, ca - cb
    if not add and not rem:
        return "Đảo vị trí thực thể" if a != b else "Giống hệt"
    if (NEG & set(add)) or (NEG & set(rem)):
        return "Thêm/bớt phủ định"
    if any(re.search(r"\d", t) for t in list(add) + list(rem)):
        return "Đổi số / thời gian"
    if any((x in add and y in rem) or (y in add and x in rem) for x, y in ANTONYMS):
        return "Đổi sang từ trái nghĩa"
    caps = {w.lower().strip("?,.") for w in (imp_q + " " + sib_q).split() if w[:1].isupper()}
    if any(t in caps for t in list(add) + list(rem)):
        return "Thay thực thể (tên riêng)"
    return "Thay từ/cụm thường"


def is_letter(ch):
    return bool(ch) and unicodedata.category(ch).startswith("L")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def summ(xs):
    return {"mean": round(st.mean(xs), 2), "median": st.median(xs), "p05": pct(xs, .05),
            "p95": pct(xs, .95), "max": max(xs), "n": len(xs)}


def hist(xs, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append({"bin": f"{lo}-{hi - 1}" if hi < 10**6 else f"{lo}+",
                    "count": sum(lo <= x < hi for x in xs)})
    return out


def load(split):
    return json.load(open(RAW / f"viquad2_{split}.json", encoding="utf-8"))["data"]


def iter_q(data):
    for art in data:
        for pi, p in enumerate(art["paragraphs"]):
            for q in p["qas"]:
                yield art["title"], pi, p, q


def analyse_split(split):
    data = load(split)
    labelled = split != "test"
    r = {"articles": len(data), "paragraphs": sum(len(a["paragraphs"]) for a in data)}
    ctx_len, q_len, a_len, a_rel, imp, n = [], [], [], [], 0, 0
    qt, at_, qt_imp = Counter(), Counter(), Counter()
    q_ctx_overlap = {"answerable": [], "impossible": []}
    ans_in_q, sent_rank1, sent_total, sent_idx = 0, 0, 0, []
    noise = Counter()
    ngold = Counter()
    per_article = []
    imp_sim, plaus_is_sibling, plaus_in_ctx = [], 0, 0
    added, removed = Counter(), Counter()
    edits = Counter()
    edit_examples = defaultdict(list)
    for art in data:
        aq = ai = 0
        for p in art["paragraphs"]:
            ctx = p["context"]
            ctx_len.append(len(ctx.split()))
            ctx_toks = set(tokenize(ctx))
            sents = sentences(ctx)
            ans_qs = [q for q in p["qas"] if not q.get("is_impossible")]
            sib_answers = {tokenize(t) and " ".join(tokenize(t))
                           for q in ans_qs for t in q["answers"]["text"]}
            for q in p["qas"]:
                n += 1
                aq += 1
                qtok = tokenize(q["question"])
                q_len.append(len(q["question"].split()))
                t = qtype(q["question"])
                qt[t] += 1
                ov = sum(tok in ctx_toks for tok in qtok) / max(1, len(qtok))
                if not labelled:
                    continue
                if q.get("is_impossible"):
                    imp += 1
                    ai += 1
                    qt_imp[t] += 1
                    q_ctx_overlap["impossible"].append(ov)
                    best, best_q = 0.0, None
                    for s in ans_qs:
                        j = jacc(qtok, tokenize(s["question"]))
                        if j > best:
                            best, best_q = j, s
                    imp_sim.append(best)
                    if best_q is not None:
                        et = edit_type(q["question"], best_q["question"])
                        edits[et] += 1
                        if len(edit_examples[et]) < 3 and best >= .6:
                            edit_examples[et].append({"answerable": best_q["question"], "impossible": q["question"],
                                                      "plausible": q.get("plausible_answers", {}).get("text", [""])[0]})
                    if best_q is not None and best >= .5:
                        sq = Counter(tokenize(best_q["question"]))
                        iq = Counter(qtok)
                        added.update((iq - sq).keys())
                        removed.update((sq - iq).keys())
                    pa = q.get("plausible_answers", {}).get("text", [])
                    if pa:
                        if " ".join(tokenize(pa[0])) in sib_answers:
                            plaus_is_sibling += 1
                        s0 = q["plausible_answers"]["answer_start"][0]
                        if ctx[s0:s0 + len(pa[0])] == pa[0]:
                            plaus_in_ctx += 1
                    continue
                q_ctx_overlap["answerable"].append(ov)
                texts, starts = q["answers"]["text"], q["answers"]["answer_start"]
                ngold[len(set(texts))] += 1
                a, s0 = texts[0], starts[0]
                a_len.append(len(a.split()))
                a_rel.append(s0 / max(1, len(ctx)))
                at_[atype(a)] += 1
                atoks = tokenize(a)
                if atoks and sum(x in set(qtok) for x in atoks) / len(atoks) >= .5:
                    ans_in_q += 1
                # nhiễu nhãn: đáp án bắt đầu / kết thúc giữa âm tiết
                if ctx[s0:s0 + len(a)] != a:
                    noise["offset_mismatch"] += 1
                else:
                    if s0 > 0 and is_letter(ctx[s0 - 1]) and is_letter(a[:1]):
                        noise["cut_start"] += 1
                    e = s0 + len(a)
                    if e < len(ctx) and is_letter(ctx[e]) and is_letter(a[-1:]):
                        noise["cut_end"] += 1
                # câu chứa đáp án có phải câu trùng từ với câu hỏi nhiều nhất?
                idx = next((i for i, (b, e, _) in enumerate(sents) if b <= s0 < e), None)
                if idx is not None and len(sents) > 1:
                    scores = [len(set(tokenize(s)) & set(qtok)) for *_, s in sents]
                    sent_total += 1
                    sent_rank1 += scores[idx] == max(scores)
                    sent_idx.append(min(idx, 6))
        per_article.append({"title": art["title"], "questions": aq,
                            "impossible_pct": round(100 * ai / aq, 1) if labelled else None})
    r.update({"questions": n, "questions_per_paragraph": round(n / r["paragraphs"], 2),
              "context_words": summ(ctx_len), "question_words": summ(q_len),
              "context_hist": hist(ctx_len, [0, 100, 150, 200, 250, 300, 400, 10**7]),
              "question_type": dict(qt.most_common()), "per_article": per_article})
    if not labelled:
        r["note"] = "private test VLSP 2021 — không có nhãn (mọi is_impossible=false, answers rỗng)"
        return r
    na = n - imp
    r.update({
        "impossible": imp, "impossible_pct": round(100 * imp / n, 2),
        "question_type_impossible_pct": {k: round(100 * qt_imp[k] / v, 1) for k, v in qt.items()},
        "answer_words": summ(a_len),
        "answer_hist": hist(a_len, [1, 2, 3, 6, 11, 21, 10**7]),
        "answer_rel_position": {"q1": round(pct(a_rel, .25), 2), "median": round(pct(a_rel, .5), 2),
                                "q3": round(pct(a_rel, .75), 2),
                                "deciles": [sum(i / 10 <= x < (i + 1) / 10 or (i == 9 and x >= 1)
                                                for x in a_rel) for i in range(10)]},
        "answer_type": dict(at_.most_common()),
        "gold_variants": dict(sorted(ngold.items())),
        "question_context_overlap": {k: round(100 * st.mean(v), 1) for k, v in q_ctx_overlap.items()},
        "answer_mostly_in_question_pct": round(100 * ans_in_q / na, 2),
        "answer_sentence_is_top_overlap_pct": round(100 * sent_rank1 / sent_total, 1),
        "answer_sentence_index": dict(sorted(Counter(sent_idx).items())),
        "label_noise": {k: v for k, v in noise.items()} | {
            "cut_any_pct": round(100 * (noise["cut_start"] + noise["cut_end"]) / na, 2)},
        "impossible_construction": {
            "max_jaccard_to_sibling": summ([round(x, 3) for x in imp_sim]),
            "sibling_jaccard_ge_0.5_pct": round(100 * sum(x >= .5 for x in imp_sim) / imp, 1),
            "sibling_jaccard_ge_0.7_pct": round(100 * sum(x >= .7 for x in imp_sim) / imp, 1),
            "sim_hist": hist([int(100 * x) for x in imp_sim], [0, 30, 50, 70, 90, 101]),
            "plausible_equals_sibling_answer_pct": round(100 * plaus_is_sibling / imp, 1),
            "plausible_offset_ok_pct": round(100 * plaus_in_ctx / imp, 1),
            "edit_type": dict(edits.most_common()),
            "edit_examples": edit_examples,
            "top_added_tokens": added.most_common(25),
            "top_removed_tokens": removed.most_common(25),
        },
    })
    return r


def model_slices():
    """Hiệu năng của 2 mô hình (τ chọn trên dev) theo các lát cắt đặc trưng dữ liệu."""
    data = load("validation")
    refs, meta = {}, {}
    for title, pi, p, q in iter_q(data):
        qid = q["id"]
        refs[qid] = [] if q.get("is_impossible") else q["answers"]["text"]
        m = {"qtype": qtype(q["question"]), "article": title}
        qtok = tokenize(q["question"])
        if q.get("is_impossible"):
            sibs = [s for s in p["qas"] if not s.get("is_impossible")]
            sims = [jacc(qtok, tokenize(s["question"])) for s in sibs]
            best = max(sims, default=0)
            if sibs:
                m["edit"] = edit_type(q["question"], sibs[sims.index(best)]["question"])
            m["imp_sim"] = ">=0.7" if best >= .7 else ("0.5-0.7" if best >= .5 else "<0.5")
        else:
            a = q["answers"]["text"][0]
            m["atype"] = atype(a)
            m["ngold"] = min(len(set(q["answers"]["text"])), 3)
            sents = sentences(p["context"])
            s0 = q["answers"]["answer_start"][0]
            idx = next((i for i, (b, e, _) in enumerate(sents) if b <= s0 < e), None)
            if idx is not None:
                sc = [len(set(tokenize(s)) & set(qtok)) for *_, s in sents]
                m["lexical"] = "câu đáp án trùng từ nhiều nhất" if sc[idx] == max(sc) else "không phải câu trùng nhiều nhất"
        meta[qid] = m
    out = {}
    for run, label in MODELS.items():
        f = RES / f"predictions_{run}_tuned_validation.json"
        preds = json.load(open(f, encoding="utf-8"))
        preds = preds.get("predictions", preds)
        sl = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
        for qid, gold in refs.items():
            pr = preds.get(qid, "")
            em = metric_max_over_ground_truths(exact_match, pr, gold)
            f1 = metric_max_over_ground_truths(token_f1, pr, gold)
            m = meta[qid]
            keys = [("question_type", m["qtype"]), ("article", m["article"])]
            if gold:
                keys += [("answer_type", m["atype"]), ("gold_variants", str(m["ngold"])),
                         ("lexical", m.get("lexical", "?"))]
            else:
                keys += [("impossible_similarity", m["imp_sim"]), ("impossible_edit", m.get("edit", "?"))]
            keys.append(("all", "all"))
            for k in keys:
                c = sl[k[0]][k[1]]
                c[0] += 1; c[1] += em; c[2] += f1
        out[run] = {g: {k: {"n": c[0], "EM": round(100 * c[1] / c[0], 2), "F1": round(100 * c[2] / c[0], 2)}
                        for k, c in d.items()} for g, d in sl.items()}
        out[run]["label"] = label
    return out


def windows():
    out = {}
    for run in MODELS:
        d = json.load(open(RES / f"windows_{run}_validation.json", encoding="utf-8"))["records"]
        c = Counter(min(len(v), 4) for v in d.values())
        out[run] = {"windows_per_question": {("4+" if k == 4 else str(k)): v for k, v in sorted(c.items())},
                    "mean": round(st.mean(len(v) for v in d.values()), 3)}
    return out


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    res = {"commit": commit,
           "splits": {s: analyse_split(s) for s in ("train", "validation", "test")},
           "windows": windows(), "model_slices": model_slices()}
    t = {a["title"] for a in load("train")}
    v = {a["title"] for a in load("validation")}
    res["title_overlap_train_validation"] = sorted(t & v)
    (RES / "dataset_analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote results/dataset_analysis.json")


if __name__ == "__main__":
    main()
