"""Năm trang của ViMRC Console, dựng đúng theo bản thiết kế.

Mỗi khối nội dung được sinh thành HTML rồi đưa vào Streamlit, vì bản thiết kế dùng
grid, ô nhiệt và thanh xếp lớp mà widget mặc định của Streamlit không dựng được.
Phần tương tác (ô nhập, nút, chip) là widget Streamlit đã được CSS trong
``theme.py`` bọc lại cho khớp.

Số liệu lấy từ ``data.py``, mô hình từ ``infer.py``; ở đây không có con số nào
viết tay.
"""
from __future__ import annotations

import html

import streamlit as st

from . import data as D
from . import infer
from . import markup as M
from .theme import heat

MISSING = "<span class='vm-muted'>chưa có dữ liệu trong results/</span>"

_ERR_LABELS = {
    "false_answer": "trả lời câu không có đáp án", "false_abstain": "từ chối sai",
    "boundary_superset": "biên thừa", "boundary_subset": "biên thiếu",
    "boundary_overlap": "chồng lấn một phần", "wrong_span": "sai hẳn",
}


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _err_label(key) -> str:
    return _ERR_LABELS.get(str(key), str(key))


def _chips(options: list[tuple[str, str]], state_key: str, ncols: int | None = None,
           per_row: int | None = None) -> str:
    """Chip chọn-một, trả về khoá đang chọn. Chip = nút pill của theme.py.

    ``per_row`` giới hạn số chip mỗi dòng: cột Streamlit không tự xuống dòng, nên
    nhãn dài trong cột hẹp sẽ bị cắt nếu nhồi hết vào một dòng.
    """
    current = st.session_state.get(state_key, options[0][0])
    width = per_row or ncols or len(options)
    for start in range(0, len(options), width):
        chunk = options[start:start + width]
        cols = st.columns(ncols or width) if per_row is None else st.columns(width)
        for (key, label), col in zip(chunk, cols):
            with col:
                if st.button(label, key=f"{state_key}-{key}", use_container_width=True,
                             type="primary" if key == current else "secondary"):
                    st.session_state[state_key] = key
                    current = key
                    st.rerun()
    return current


def _highlight(context: str, marks: list[tuple[str, str]]) -> str:
    """Bọc <mark> quanh đáp án vàng và đáp án dự đoán, an toàn với HTML."""
    if not context:
        return MISSING
    parts: list = [context]
    for text, cls in marks:
        if not text:
            continue
        nxt: list = []
        for chunk in parts:
            if not isinstance(chunk, str):
                nxt.append(chunk)
                continue
            for i, piece in enumerate(chunk.split(text)):
                if i:
                    nxt.append((text, cls))
                if piece:
                    nxt.append(piece)
        parts = nxt
    out = []
    for p in parts:
        if isinstance(p, tuple):
            out.append(f"<mark class='{p[1]}'>{esc(p[0])}</mark>")
        else:
            out.append(esc(p))
    return "".join(out)


# ── 1. Hỏi đáp ──────────────────────────────────────────────────────────────
def qa_page() -> None:
    runs = infer.available_runs()
    if not runs:
        st.warning("Chưa có checkpoint nào trong `models/`. Huấn luyện trước bằng "
                   "`python scripts/finetune.py …` rồi mở lại trang này. "
                   "Bốn trang còn lại vẫn đọc được từ `results/`.")
        return

    examples = st.cache_data(infer.examples)()
    if examples:
        _md("<div class='vm-label' style='margin-bottom:6px'>Câu mẫu — chọn theo cơ chế lỗi"
            "</div>")
        picked = _chips([(e["qid"], e["label"]) for e in examples], "qa-example")
        ex = next((e for e in examples if e["qid"] == picked), examples[0])
    else:
        ex = {"qid": "—", "context": "", "question": "", "gold": []}

    if st.session_state.get("qa-loaded") != ex["qid"]:
        st.session_state["qa-context"] = ex["context"]
        st.session_state["qa-question"] = ex["question"]
        st.session_state["qa-loaded"] = ex["qid"]

    left, right = st.columns(2, gap="medium")
    with left:
        context = st.text_area("Ngữ cảnh", key="qa-context", height=240)
        question = st.text_input("Câu hỏi", key="qa-question")
        _md(f"<div class='vm-muted' style='font-size:11.5px;margin:-6px 0 10px'>"
            f"<span class='vm-mono'>{esc(ex['qid'])}</span> · {len(context.split())} âm tiết · "
            f"cửa sổ 256 token, stride 96</div>")
        run_key = _chips([(r["key"], r["label"]) for r in runs], "qa-run", per_row=2)
        tuned = _chips([("tuned", "τ chọn trên dev"), ("zero", "τ = 0")], "qa-tau")
        tau = infer.tau_for(run_key) if tuned == "tuned" else 0.0
        go = st.button("Chạy dự đoán", type="primary", key="qa-run-btn")

    if not (go or st.session_state.get("qa-has-run")):
        with right:
            _md("<div class='vm-card'><div class='vm-label'>Kết quả</div>"
                "<p class='vm-lede' style='margin-top:8px'>Chọn một câu mẫu hoặc dán ngữ cảnh của "
                "bạn, rồi bấm <strong>Chạy dự đoán</strong>. Mô hình chạy trên máy này, cùng lớp "
                "inference và cùng ngưỡng τ mà bảng kết quả đã dùng.</p></div>")
        return
    st.session_state["qa-has-run"] = True

    if not context.strip() or not question.strip():
        with right:
            st.warning("Cần cả ngữ cảnh và câu hỏi. Chọn một câu mẫu ở trên hoặc dán "
                       "đoạn văn của bạn rồi nhập câu hỏi.")
        return

    # Nhãn vàng chỉ có nghĩa khi câu hỏi chưa bị sửa khỏi câu mẫu.
    gold = ex["gold"] if question.strip() == str(ex["question"]).strip() else []
    try:
        predictor = st.cache_resource(infer.load_predictor)(run_key)
        with st.spinner("Đang chạy mô hình trên máy này…"):
            out = infer.predict(predictor, context, question, tau)
    except Exception as exc:  # thiếu checkpoint, hết bộ nhớ, …
        with right:
            st.error("Không chạy được mô hình trên máy này. Kiểm tra checkpoint trong "
                     "`models/` và bộ nhớ còn trống, rồi thử lại.")
            with st.expander("Chi tiết kỹ thuật"):
                st.code(f"{type(exc).__name__}: {exc}")
        return

    em, f1 = infer.score(out["answer"], gold)
    good = em >= 0.5
    with right:
        bg = "var(--color-accent-2-200)" if good else "var(--color-accent-200)"
        fg = "var(--color-accent-2-900)" if good else "var(--color-accent-900)"
        answer = esc(out["answer"]) if out["answer"] else "⌀&nbsp; mô hình từ chối trả lời"
        tag = "đúng" if good else ("từ chối" if not out["answer"] else "sai")
        stats = [
            ("Nhãn vàng", esc(gold[0]) if gold else "⌀ không có đáp án"),
            ("EM", D.vi(em * 100, 0)),
            ("F1", D.vi(f1 * 100, 1)),
            ("Biên độ so với null", D.vi(out["null_delta"])),
            ("τ đang dùng", D.vi(tau)),
        ]
        stats_html = "".join(f"<div><div class='vm-stat-k'>{k}</div>"
                             f"<div class='vm-stat-v'>{v}</div></div>" for k, v in stats)
        _md(f"""<div class='vm-verdict' style='background:{bg};color:{fg}'>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>
            <span class='vm-kicker' style='color:{fg}'>
              {'Khớp nhãn vàng' if good else 'Lệch nhãn vàng'}</span>
            <span class='vm-tag {"vm-tag-ok" if good else "vm-tag-err"}'>{tag}</span>
          </div>
          <div class='vm-verdict-answer'>{answer}</div>
          <div class='vm-stats'>{stats_html}</div>
        </div>""")

        marks = []
        if gold:
            marks.append((gold[0], "vm-mark-gold"))
        if out["answer"] and out["answer"] != (gold[0] if gold else None):
            marks.append((out["answer"], "vm-mark-pred"))
        _md(f"""<div class='vm-card' style='margin-top:14px'>
          <div class='vm-label' style='margin-bottom:10px'>Vị trí đáp án trong ngữ cảnh</div>
          <div class='vm-ctx'>{_highlight(context, marks)}</div>
          <div class='vm-legend'>
            <span><span class='vm-swatch' style='background:var(--color-accent-2-300)'></span>
              đáp án vàng</span>
            <span><span class='vm-swatch' style='background:var(--color-accent-300)'></span>
              mô hình dự đoán</span>
          </div></div>""")

    # Các nhánh mô hình trả lời gì — chạy mọi checkpoint có sẵn trên cùng đầu vào.
    rows = []
    for r in runs:
        try:
            p = st.cache_resource(infer.load_predictor)(r["key"])
            o = infer.predict(p, context, question,
                              infer.tau_for(r["key"]) if tuned == "tuned" else 0.0)
        except Exception:
            continue
        e, _ = infer.score(o["answer"], gold)
        ok = e >= 0.5
        rows.append(
            f"<div class='vm-side' style='background:"
            f"{'var(--color-accent-2-100)' if ok else 'var(--color-accent-100)'}'>"
            f"<span style='font-size:12.5px;font-weight:700'>{esc(r['label'])}</span>"
            f"<span style='font-size:13.5px'>{esc(o['answer']) or '⌀ từ chối trả lời'}</span>"
            f"<span class='vm-tag {'vm-tag-ok' if ok else 'vm-tag-err'}'>"
            f"{'đúng' if ok else 'sai'}</span></div>")
    ab_ok = not gold
    rows.append(
        f"<div class='vm-side' style='background:"
        f"{'var(--color-accent-2-100)' if ab_ok else 'var(--color-accent-100)'}'>"
        f"<span style='font-size:12.5px;font-weight:700'>Luôn từ chối</span>"
        f"<span style='font-size:13.5px'>⌀ từ chối trả lời</span>"
        f"<span class='vm-tag {'vm-tag-ok' if ab_ok else 'vm-tag-err'}'>"
        f"{'đúng' if ab_ok else 'sai'}</span></div>")
    _md(f"<div class='vm-card' style='margin-top:14px'>"
        f"{M.heading('Các nhánh mô hình trả lời gì')}{''.join(rows)}</div>")

    seg, causes = D.seg_summary(), D.seg_causes()
    if seg and causes:
        cards = "".join(
            f"<div class='vm-cause'>"
            f"<div style='display:flex;align-items:baseline;justify-content:space-between;gap:8px'>"
            f"<span style='font-size:13px;font-weight:700'>{esc(c['name'])}</span>"
            f"<span style='font-weight:800;font-size:22px;color:var(--color-accent-700)'>"
            f"{c['n']}</span></div>"
            f"<div class='vm-muted' style='font-size:12.5px'>{esc(c['desc'])}</div></div>"
            for c in causes)
        _md(f"""<div class='vm-card' style='margin-top:14px'>
          <div style='display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:10px'>
            {M.heading('Ranh giới từ ghép trên tập validation', style='margin:0')}
            <span class='vm-muted' style='font-size:13px'>{D.vi(seg['aligned'], 0)} /
              {D.vi(seg['answerable'], 0)} đáp án khớp ranh giới pyvi — {D.vi(seg['aligned_pct'])}%.
              Trần EM do tách từ: {D.vi(seg['oracle'])}%.</span>
          </div>
          <div class='vm-causes' style='margin-top:14px'>{cards}</div></div>""")


# ── 2. Ma trận chẩn đoán ────────────────────────────────────────────────────
def matrix_page(system_key: str) -> None:
    tiles = D.tiles(system_key)
    if tiles:
        def dfg(t):
            if t["good"] is None:
                return "var(--color-neutral-700)"
            return "var(--color-accent-2-700)" if t["good"] else "var(--color-accent-700)"

        _md("<div class='vm-tiles'>" + "".join(
            f"<div class='vm-tile'><span class='vm-tile-k'>{esc(t['k'])}</span>"
            f"<span class='vm-tile-v'>{esc(t['v'])}</span>"
            f"<span class='vm-tile-d' style='color:{dfg(t)}'>{esc(t['delta'])}</span></div>"
            for t in tiles) + "</div>")

    rows = D.main_rows()
    if rows:
        head = "".join(f"<th>{esc(c)}</th>" for c in D.MAIN_COLS)
        body = "".join(
            f"<tr class='{'vm-row-active' if r['key'] == system_key else ''}'>"
            f"<td>{esc(r['label'])}</td>"
            + "".join(f"<td style='color:"
                      f"{'var(--color-text)' if i < 2 else 'var(--color-neutral-700)'};"
                      f"font-weight:{700 if i < 2 else 400}'>{D.vi(v)}</td>"
                      for i, v in enumerate(r["cells"])) + "</tr>" for r in rows)
        n = D.dig(D.load("eval_phobert_validation.json") or {}, "overall", "count")
        _md(f"""<div class='vm-card' style='margin-top:14px'>
          {M.heading(f"Kết quả chính — UIT-ViQuAD 2.0 validation (n = {D.vi(n, 0)})")}
          <div class='vm-scroll'><table class='vm-table'><thead><tr><th>Hệ thống</th>{head}</tr></thead>
          <tbody>{body}</tbody></table></div>
          <p class='vm-muted' style='margin:14px 0 0;font-size:12.5px;max-width:90ch'>
            Ở τ = 0, PhoBERT seed 42 đạt EM có-đáp-án cao nhất nhưng gần như không từ chối, nên EM
            không-đáp-án rất thấp; seed 13 đảo lại cân bằng đó. Cột “tỉ lệ từ chối” là chìa khoá
            để đọc bảng này.</p></div>""")

    _md(M.heading("Ma trận chẩn đoán E1–E5 (EM, bộ stress-test v2)",
                  style="margin:22px 0 10px"))
    scope = _chips([("category", "Theo nhóm E1–E5"), ("subset", "Theo tập con")],
                   "matrix-scope", ncols=4)
    cols, srows = D.stress_matrix(scope)
    if srows:
        head = "".join(f"<th>{esc(c['code'])}<br><span>{esc(c['name'])}</span></th>" for c in cols)
        body = ""
        for r in srows:
            cells = ""
            for v in r["cells"]:
                bg, fg = heat(v)
                cells += f"<td style='background:{bg};color:{fg}'>{D.vi(v, 1)}</td>"
            body += f"<tr><td>{esc(r['short'])}</td>{cells}</tr>"
        _md(f"""<div class='vm-card'>
          <div class='vm-scroll'><table class='vm-matrix'><thead><tr><th>Hệ thống</th>{head}</tr></thead>
          <tbody>{body}</tbody></table></div>
          <p class='vm-muted' style='margin:14px 0 0;font-size:12.5px;max-width:90ch'>
            {esc(D.stress_note(scope))}</p></div>""")

    tax = D.taxonomy_rows()
    if tax:
        rows_html = ""
        for r in tax:
            rows_html += (f"<div class='vm-taxrow'><span style='font-size:13px;font-weight:700'>"
                          f"{esc(r['short'])}</span>"
                          f"{M.bar(r['short'], r['segs'], r['total'])}"
                          f"<span class='vm-num vm-muted' style='font-size:13px;text-align:right'>"
                          f"{D.vi(r['total'], 0)}</span></div>")
        legend = "".join(
            f"<span><span class='vm-swatch' style='background:{bg}'></span>{esc(label)}</span>"
            for _, label, bg in D.TAX_TYPES)
        _md(f"""<div class='vm-card' style='margin-top:14px'>
          {M.heading('Phân loại lỗi — hình dạng thất bại, không phải số lượng')}
          <p class='vm-muted' style='margin:0 0 14px;font-size:13px'>Mỗi thanh là toàn bộ số câu
            sai của một hệ thống, chia theo cơ chế.</p>
          {rows_html}<div class='vm-legend'>{legend}</div></div>""")


# ── 3. Khám phá lỗi ─────────────────────────────────────────────────────────
def errors_page() -> None:
    kind = _chips(list(D.ERROR_FILTERS), "err-filter", ncols=4)
    items, total = D.disagreements(kind)
    _md(f"<div class='vm-muted' style='font-size:12.5px;margin:6px 0 12px'>"
        f"hiện {len(items)} / {D.vi(total, 0)} câu khớp bộ lọc</div>")

    for e in items:
        gold = e.get("gold") or []
        gold_txt = " · ".join(gold) if gold else "⌀ không có đáp án"
        winner = "PhoBERT" if e.get("winner") == "phobert" else "XLM-R"
        win_bg = "var(--color-accent-700)" if winner == "PhoBERT" else "var(--color-accent-2-700)"
        preds = ""
        for name, key, err_key in (("PhoBERT", "phobert", "phobert_error"),
                                   ("XLM-R", "xlmr", "xlmr_error")):
            ok = not e.get(err_key)
            preds += (
                f"<div class='vm-kv' style='background:"
                f"{'var(--color-accent-2-100)' if ok else 'var(--color-accent-100)'}'>"
                f"<span style='font-size:12.5px;font-weight:700'>{name}</span>"
                f"<span style='font-size:14px'>{esc(e.get(key)) or '⌀ từ chối trả lời'}</span>"
                f"<span class='vm-tag {'vm-tag-ok' if ok else 'vm-tag-err'}'>"
                f"{'đúng' if ok else esc(_err_label(e.get(err_key)))}</span></div>")
        _md(f"""<div class='vm-err'>
          <div class='vm-err-head'>
            <span class='vm-mono vm-muted' style='font-size:12px'>{esc(e.get('qid'))}</span>
            <span class='vm-err-q'>{esc(e.get('question'))}</span>
            <span class='vm-tag' style='margin-left:auto;background:{win_bg};color:#fff'>
              {winner} thắng</span>
          </div>
          <div class='vm-kv' style='background:transparent;padding-left:0'>
            <span class='vm-label'>Đáp án vàng</span>
            <span style='font-size:14px;color:var(--color-accent-2-800)'>{esc(gold_txt)}</span>
            <span></span>
          </div>{preds}</div>""")

    _md(f"<p class='vm-muted' style='font-size:12.5px;max-width:90ch'>{esc(D.paired_summary())}</p>")


# ── 4. Bộ stress-test ──────────────────────────────────────────────────────
def stress_page() -> None:
    tiles = D.audit_tiles()
    if tiles:
        _md("<div class='vm-tiles'>" + "".join(
            f"<div class='vm-tile'><span class='vm-tile-k'>{esc(t['k'])}</span>"
            f"<span class='vm-tile-v'>{esc(t['v'])}</span>"
            f"<span class='vm-tile-d vm-muted'>{esc(t['note'])}</span></div>"
            for t in tiles) + "</div>")

    groups = D.group_cards()
    if not groups:
        _md(MISSING)
        return
    _md("<div class='vm-label' style='margin:18px 0 6px'>Chọn một nhóm để xem kiểm định và mẫu</div>")
    code = _chips([(g["code"], f"{g['code']} · {g['name']}") for g in groups],
                  "stress-group", ncols=5)
    g = next((x for x in groups if x["code"] == code), groups[0])

    _md("<div class='vm-tiles' style='margin-bottom:14px'>" + "".join(
        f"<div class='vm-tile'><span class='vm-tile-k'>{esc(k)}</span>"
        f"<span class='vm-tile-v' style='font-size:26px'>{esc(v)}</span>"
        f"<span class='vm-tile-d vm-muted'>{esc(note)}</span></div>"
        for k, v, note in (
            ("Mục", f"{g['n']}", "đều chấm được — 0 vi phạm kiểm định"),
            ("Có đáp án", f"{g['ans']}/{g['n']}",
             f"{g['imp']} câu không có đáp án"),
            ("Câu gốc đi kèm", f"{g['twins']}", "để so theo cặp trước/sau biến đổi"),
        )) + "</div>")

    warn = D.group_warning(code)
    if warn:
        _md(f"<div class='vm-groupwarn'><span style='font-weight:800'>⚠</span>"
            f"<p style='margin:0'>{esc(warn)}</p></div>")

    samples = "".join(
        f"<div class='vm-samples'>"
        f"<span class='vm-mono vm-muted' style='font-size:12px'>{esc(s['qid'])}</span>"
        f"<span style='font-size:13.5px'>{esc(s['question'])}</span>"
        f"<span style='font-size:13px;color:var(--color-accent-2-800)'>{esc(s['gold'])}</span>"
        f"<span class='vm-tag vm-tag-ok'>"
        f"{esc(s['subset'])}{' · ' + esc(s['role']) if s['role'] else ''}</span></div>"
        for s in D.group_samples(code))
    _md(f"<div class='vm-card' style='margin-top:14px'>"
        f"{M.heading(g['code'] + ' — ' + g['name'])}"
        f"<p class='vm-muted' style='font-size:13px;margin:0 0 14px'>"
        f"{esc(g['mech'])}</p>{samples or MISSING}</div>")


# ── 5. Huấn luyện & ngưỡng ─────────────────────────────────────────────────
def runs_page() -> None:
    rows = D.run_rows()
    if rows:
        _md(f"<div class='vm-card'>{M.heading('F1 trên tập dev theo epoch')}"
            f"{_curve_svg(rows)}</div>")

        head = "".join(f"<th>{esc(c)}</th>" for c in D.RUN_COLS)
        body = ""
        for r in rows:
            cells = ""
            for i, v in enumerate(r["cells"]):
                if i == 4:
                    fg = ("var(--color-neutral-700)" if not r["has_stability"]
                          else ("var(--color-accent-2-700)" if r["stable"]
                                else "var(--color-accent-700)"))
                elif i in (3, 5):
                    fg = "var(--color-text)"
                else:
                    fg = "var(--color-neutral-700)"
                cells += (f"<td style='color:{fg};font-weight:{700 if i in (3, 4, 5) else 400}'>"
                          f"{esc(v)}</td>")
            body += f"<tr><td>{esc(r['name'])}</td>{cells}</tr>"
        _md(f"""<div class='vm-card' style='margin-top:14px'>
          {M.heading('Các lần huấn luyện')}
          <div class='vm-scroll'><table class='vm-table'><thead><tr><th>Lần chạy</th>{head}</tr></thead>
          <tbody>{body}</tbody></table></div>
          <p class='vm-muted' style='margin:14px 0 0;font-size:12px'>* Lần chạy trước khi có nhật ký
            từng bước: trạng thái nhánh CLS suy ra từ loss theo nhãn của từng checkpoint
            (<span class='vm-mono'>results/loss_probe.json</span>), không phải từ chuẩn gradient
            từng bước.</p></div>""")

    thr = D.threshold_rows()
    if thr:
        head = "".join(f"<th>{esc(c)}</th>" for c in D.THR_COLS)
        body = ""
        for r in thr:
            cells = ""
            for i, v in enumerate(r["cells"]):
                fg = ("var(--color-accent-700)" if i == 0
                      else ("var(--color-text)" if i == 2 else "var(--color-neutral-700)"))
                cells += (f"<td style='color:{fg};font-weight:{700 if i in (0, 2) else 400}'>"
                          f"{esc(v)}</td>")
            body += f"<tr><td>{esc(r['name'])}</td>{cells}</tr>"
        _md(f"""<div class='vm-card' style='margin-top:14px'>
          <div style='display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:10px'>
            {M.heading('Ngưỡng từ chối τ — chọn trên dev, áp lên validation', style='margin:0')}
            <span class='vm-muted' style='font-size:12.5px'>τ dịch cân bằng có/không đáp án mà
              không huấn luyện lại</span>
          </div>
          <div class='vm-scroll'><table class='vm-table' style='margin-top:14px'>
          <thead><tr><th>Lần chạy</th>{head}</tr></thead><tbody>{body}</tbody></table></div>
          <p class='vm-muted' style='margin:14px 0 0;font-size:12.5px;max-width:90ch'>
            {esc(D.threshold_note())}</p></div>""")


def _curve_svg(rows: list[dict]) -> str:
    """Đường cong F1 dev theo epoch — SVG thuần, thang tự co theo dữ liệu thật."""
    pts = [v for r in rows for v in r["curve"]]
    if not pts:
        return MISSING
    lo, hi = int(min(pts)) - 2, int(max(pts)) + 2
    left, right, top, bot = 46, 545, 16, 216
    n = max(len(r["curve"]) for r in rows)
    step = max((hi - lo) // 5, 1)

    def x(i: float) -> float:
        return left + (i * (right - left) / max(n - 1, 1))

    def y(v: float) -> float:
        return bot - ((v - lo) / (hi - lo)) * (bot - top)

    grid = "".join(f"<line x1='{left}' y1='{y(v):.1f}' x2='{right}' y2='{y(v):.1f}'/>"
                   for v in range(lo, hi + 1, step))
    labels = "".join(f"<text x='38' y='{y(v) + 4:.1f}'>{v}</text>"
                     for v in range(lo, hi + 1, step))
    ticks = "".join(f"<text x='{x(i):.1f}' y='242'>epoch {i + 1}</text>" for i in range(n))
    lines, legend = "", ""
    for r in rows:
        if not r["curve"]:
            continue
        poly = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(r["curve"]))
        lines += (f"<polyline points='{poly}' fill='none' stroke='{r['color']}' "
                  f"stroke-width='2.6' stroke-linecap='round' stroke-linejoin='round'/>"
                  f"<circle cx='{x(len(r['curve']) - 1):.1f}' cy='{y(r['curve'][-1]):.1f}' "
                  f"r='4.5' fill='{r['color']}'/>")
        legend += (f"<div style='display:flex;align-items:baseline;gap:8px;font-size:13px'>"
                   f"<span style='width:16px;height:4px;border-radius:999px;flex:none;"
                   f"background:{r['color']}'></span>"
                   f"<span style='flex:1'>{esc(r['curve_label'])}</span>"
                   f"<span class='vm-num' style='font-weight:700'>{D.vi(r['curve'][-1])}</span></div>")
    return (f"<div style='display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);"
            f"gap:20px;align-items:center'>"
            f"<svg viewBox='0 0 560 250' style='width:100%;height:auto;overflow:visible'>"
            f"<g stroke='var(--color-neutral-300)' stroke-width='1'>{grid}</g>"
            f"<g text-anchor='end' font-size='11' fill='#645c50'>{labels}</g>"
            f"<g text-anchor='middle' font-size='11' fill='#645c50'>{ticks}</g>{lines}</svg>"
            f"<div style='display:flex;flex-direction:column;gap:8px'>{legend}</div></div>")
