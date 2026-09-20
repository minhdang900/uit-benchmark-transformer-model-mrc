"""ViMRC Console — chạy: ``streamlit run demo/app.py`` (từ gốc repo).

Khung trang theo bản thiết kế: thanh bên 292px (thương hiệu · điều hướng · chọn
hệ thống · provenance) và vùng nội dung có tiêu đề cùng năm trang.

Số liệu đọc từ ``results/``; trang Hỏi đáp chạy checkpoint thật trong ``models/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from console import data as D  # noqa: E402
from console import markup as M  # noqa: E402
from console import pages  # noqa: E402
from console.theme import css  # noqa: E402

TEAM = [
    ("Nguyễn Quang Lâm", "25210289"),
    ("Trần Trọng Tấn", "25210334"),
    ("Lê Quang Thi", "25210337"),
    ("Vỏ Cẩm Thu", "25210342"),
]

PAGES = {
    "qa": {"label": "Hỏi đáp", "note": "thử một câu, xem mô hình gãy ở đâu",
           "title": "Hỏi đáp trích xuất",
           "lede": "Chạy một cặp câu hỏi – ngữ cảnh qua các nhánh mô hình và xem ngay đáp án "
                   "lệch ở đâu so với nhãn vàng."},
    "matrix": {"label": "Ma trận chẩn đoán", "note": "E1–E5 · 6 hệ thống",
               "title": "Ma trận chẩn đoán",
               "lede": "Giá trị nằm ở hình dạng chênh lệch giữa các cột, không phải ở con số "
                       "cao nhất."},
    "errors": {"label": "Khám phá lỗi", "note": "các câu hai mô hình bất đồng",
               "title": "Khám phá lỗi",
               "lede": "Những câu mà PhoBERT và XLM-R trả lời khác nhau — nơi điểm tổng che mất "
                       "cơ chế thất bại."},
    "stress": {"label": "Bộ stress-test v2", "note": "1139 mục · 5 nhóm",
               "title": "Bộ stress-test v2",
               "lede": "Mỗi nhóm cô lập một cơ chế thất bại. Kiểm định hiện lên cùng dữ liệu, "
                       "không giấu ở phụ lục."},
    "runs": {"label": "Huấn luyện & ngưỡng", "note": "các lần chạy · τ hiệu chỉnh",
             "title": "Huấn luyện & ngưỡng",
             "lede": "Đường cong dev, tính ổn định của nhánh CLS, và ngưỡng từ chối chọn trên dev "
                     "rồi áp lên validation."},
}


def sidebar() -> tuple[str, str]:
    page = st.session_state.setdefault("page", "qa")
    system = st.session_state.setdefault("system", "phobert")

    with st.sidebar:
        st.markdown(
            "<span class='vm-sr-only'>ViMRC</span>"
            "<div class='vm-muted' style='font-size:12.5px;margin:0 0 16px'>"
            "Công cụ chẩn đoán MRC</div>", unsafe_allow_html=True)

        for key, meta in PAGES.items():
            if st.button(meta["label"], key=f"nav-{key}", use_container_width=True,
                         type="primary" if key == page else "secondary"):
                st.session_state["page"] = key
                page = key
                st.rerun()
            st.markdown(f"<div class='vm-muted' style='font-size:11.5px;margin:-6px 0 8px 18px'>"
                        f"{meta['note']}</div>", unsafe_allow_html=True)

        st.markdown("<div class='vm-kicker' style='margin:18px 0 8px'>Hệ thống đang xét</div>",
                    unsafe_allow_html=True)
        rows = {r["key"]: r for r in D.main_rows()}
        for s in D.SYSTEMS:
            cells = D.dig(rows, s["key"], "cells") or [None]
            if st.button(f"{s['short']} · {D.vi(cells[0], 1)}", key=f"sys-{s['key']}",
                         use_container_width=True,
                         type="primary" if s["key"] == system else "secondary"):
                st.session_state["system"] = s["key"]
                system = s["key"]
                st.rerun()

        st.markdown(M.team(TEAM), unsafe_allow_html=True)
        st.markdown("<div class='vm-foot'>"
                    + "".join(f"<div>{line}</div>" for line in D.provenance())
                    + "</div>", unsafe_allow_html=True)
    return page, system


def main() -> None:
    st.set_page_config(page_title="ViMRC Console", page_icon="◍", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(css(), unsafe_allow_html=True)
    st.logo(M.logo(), size='medium')

    page, system = sidebar()
    meta = PAGES[page]
    label = next((s["label"] for s in D.SYSTEMS if s["key"] == system), system)
    st.markdown(f"""
    <header style='display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;
      gap:16px;margin-bottom:22px'>
      <div style='min-width:0'>
        <h1 class='vm-h1'>{meta['title']}</h1>
        <p class='vm-lede'>{meta['lede']}</p>
      </div>
      <div style='display:flex;align-items:center;gap:8px;flex:none'>
        <span class='vm-muted' style='font-size:12px'>Đang xét</span>
        <span class='vm-tag vm-tag-accent' style='font-size:12.5px'>{label}</span>
      </div>
    </header>""", unsafe_allow_html=True)

    if page == "qa":
        pages.qa_page()
    elif page == "matrix":
        pages.matrix_page(system)
    elif page == "errors":
        pages.errors_page()
    elif page == "stress":
        pages.stress_page()
    else:
        pages.runs_page()


main()
