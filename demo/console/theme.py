"""Token và lớp thành phần của hệ thiết kế, cộng phần ghi đè cho Streamlit.

Token lấy đúng từ ``styles.css`` của hệ thiết kế (Organic). Hai chỗ khác bản gốc,
theo đúng ghi chú trong bản thiết kế: Caprasimo và Figtree không có bộ ký tự tiếng
Việt (U+1EA0–1EF9) nên giao diện tiếng Việt sẽ hiện hai kiểu chữ trong cùng một
từ; Be Vietnam Pro có đủ và đảm nhiệm cả hai vai trò.

Streamlit tự sinh phần lớn DOM, nên mọi widget phải được ghi đè bằng CSS ở đây:
nút dạng pill, ô nhập dùng nền surface, thanh bên dùng nền surface.
"""
from __future__ import annotations

TOKENS = """
:root {
  --color-bg: #f5ead8;
  --color-surface: #ebddc5;
  --color-text: #201e1d;
  --color-accent: #c67139;
  --color-accent-2: #7a8a5e;
  --color-divider: color-mix(in srgb, #201e1d 16%, transparent);

  --color-neutral-100: #f9f4ed;
  --color-neutral-200: #eee7db;
  --color-neutral-300: #dcd3c4;
  --color-neutral-400: #c0b6a5;
  --color-neutral-500: #a19786;
  --color-neutral-600: #82796a;
  --color-neutral-700: #645c50;
  --color-neutral-800: #474238;
  --color-neutral-900: #2e2b25;

  --color-accent-100: #fff2eb;
  --color-accent-200: #ffe1d0;
  --color-accent-300: #ffc6a5;
  --color-accent-400: #f6a06b;
  --color-accent-500: #d67f48;
  --color-accent-600: #b2622d;
  --color-accent-700: #8c491a;
  --color-accent-800: #643312;
  --color-accent-900: #402310;

  --color-accent-2-100: #f0fae1;
  --color-accent-2-200: #e1eecc;
  --color-accent-2-300: #ccdbb2;
  --color-accent-2-400: #aebf92;
  --color-accent-2-500: #8fa073;
  --color-accent-2-600: #728157;
  --color-accent-2-700: #56633f;
  --color-accent-2-800: #3d472b;
  --color-accent-2-900: #272e1b;

  /* Be Vietnam Pro thay Caprasimo/Figtree: hai kiểu chữ kia không có bộ ký tự
     tiếng Việt, dùng chúng thì mỗi từ hiện hai kiểu chữ. */
  --font-body: "Be Vietnam Pro", system-ui, sans-serif;
  --font-heading: "Be Vietnam Pro", system-ui, sans-serif;
  --font-heading-weight: 800;
  --font-mono: "Source Code Pro", ui-monospace, monospace;

  --space-1: 4.4px;  --space-2: 8.8px;  --space-3: 13.2px;
  --space-4: 17.6px; --space-6: 26.4px; --space-8: 35.2px;

  --radius-sm: 8px; --radius-md: 16px; --radius-lg: 28px;

  --shadow-sm: 0 1px 2px color-mix(in srgb, #2e2b25 14%, transparent);
  --shadow-md: 0 3px 10px color-mix(in srgb, #2e2b25 16%, transparent);
  --shadow-lg: 0 12px 32px color-mix(in srgb, #2e2b25 22%, transparent);
}
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,300;0,400;0,600;0,700;0,800;1,400&family=Source+Code+Pro:wght@400;600&display=swap');

__TOKENS__

/* ── khung Streamlit ─────────────────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] { background: var(--color-bg); }
[data-testid="stHeader"] { background: transparent; }
.stMainBlockContainer, [data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
  padding: var(--space-8) var(--space-8) var(--space-8);
  max-width: 1360px;
}
html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {
  font-family: var(--font-body) !important;
  color: var(--color-text);
}
/* Trừ chữ biểu tượng ra: Streamlit dùng ligature của Material Symbols, ép font chữ
   thường lên đó thì nút hiện ra chữ "keyboard_double" thay vì mũi tên. */
[data-testid="stIconMaterial"], .material-symbols-rounded, span[translate="no"] {
  font-family: "Material Symbols Rounded", "Material Icons" !important;
}
body { font-size: 15px; line-height: 1.5; }
#MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"] { display: none; }
/* Nút thu/mở thanh bên PHẢI thấy được: điều hướng chỉ nằm trong thanh bên. */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {
  visibility: visible !important;
}
[data-testid="stExpandSidebarButton"] button, [data-testid="stSidebarCollapseButton"] button {
  color: var(--color-accent-800) !important;
  background: var(--color-surface) !important;
  border: 1px solid var(--color-divider) !important;
  border-radius: 999px !important;
}

/* ── thanh bên ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--color-surface);
  border-right: 1px solid var(--color-divider);
}
@media (min-width: 901px) {
  [data-testid="stSidebar"] { width: 292px !important; }
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: var(--space-6) var(--space-4); }

/* Streamlit dành sẵn một ô trống ở đầu thanh bên cho st.logo(); đồ án không
   dùng logo nên ô đó chỉ là khoảng trắng. Bỏ ô trống và cân lại khoảng đệm.
   KHÔNG ẩn cả stSidebarHeader: nút thu gọn thanh bên nằm trong đó. */
[data-testid="stLogoSpacer"] { display: none; }
[data-testid="stSidebarLogo"], [data-testid="stHeaderLogo"],
[data-testid="stAppLogo"] { height: 26px; width: auto; margin: 0; }
/* Streamlit gán alt="Logo" cho ảnh logo và không cho đổi, nên tên ứng dụng
   được đặt riêng dưới dạng chữ chỉ dành cho trình đọc màn hình. */
.vm-sr-only { position: absolute; width: 1px; height: 1px; padding: 0;
  margin: -1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0; }
[data-testid="stSidebarHeader"] { height: auto; min-height: 0;
  padding-top: var(--space-1); padding-bottom: 0; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: var(--space-3); }
[data-testid="stSidebarUserContent"] { padding-bottom: var(--space-6); }
[data-testid="stSidebar"] .stButton > button { text-align: left; justify-content: flex-start; }

/* ── nút: pill; primary = đang chọn ─────────────────────────────────────── */
.stButton > button {
  font-family: var(--font-body) !important;
  font-weight: 600; font-size: 14px; line-height: 1.2;
  padding: 9px var(--space-4);
  border-radius: 999px;
  border: 1px solid var(--color-divider);
  background: transparent; color: var(--color-neutral-800);
  transition: none;
}
.stButton > button:hover { border-color: var(--color-accent-400); color: var(--color-text); background: transparent; }
.stButton > button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
  /* accent-700 thay vì accent: chữ trắng 14px trên accent chỉ đạt 3,61:1, dưới ngưỡng
     AA 4,5:1 cho chữ nhỏ. accent-700 đạt 6,81:1 và vẫn cùng họ màu. */
  background: var(--color-accent-700); border-color: var(--color-accent-700); color: #fff;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: var(--color-accent-800); border-color: var(--color-accent-800); color: #fff;
}
.stButton > button:active { background: color-mix(in srgb, var(--color-text) 10%, transparent); }

/* ── ô nhập ─────────────────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
  background: var(--color-surface) !important;
  border: 1px solid var(--color-divider) !important;
  border-radius: var(--radius-md) !important;
  color: var(--color-text) !important;
  font-size: 14.5px !important;
  line-height: 1.65 !important;
  caret-color: var(--color-accent);
}
.stTextInput input { border-radius: 999px !important; padding-inline: 14px !important; }
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--color-accent) !important; box-shadow: none !important;
}
.stTextInput input:focus-visible, .stTextArea textarea:focus-visible {
  outline: 2px solid var(--color-accent-700) !important; outline-offset: 2px;
}
[data-testid="stWidgetLabel"] p {
  font-size: 12.5px; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--color-neutral-700);
}

/* ── spinner, thông báo ─────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--color-accent) !important; }
[data-testid="stNotification"] { border-radius: var(--radius-md); }

/* ── lớp thành phần của hệ thiết kế ─────────────────────────────────────── */
.vm-card {
  background: var(--color-neutral-100); border-radius: var(--radius-lg);
  padding: var(--space-6); box-shadow: var(--shadow-sm);
}
.vm-card + .vm-card { margin-top: var(--space-4); }
.vm-h1 { font-weight: 800; letter-spacing: -0.015em; font-size: 38px; line-height: 1.12; margin: 0 0 6px; }
.vm-h2 { font-weight: 800; letter-spacing: -0.015em; font-size: 22px; margin: 0 0 var(--space-3); }
/* Streamlit tự đặt cỡ cho h1–h6 trong markdown với độ ưu tiên cao hơn một lớp
   đơn; tiêu đề mục là <h2> thật (để đọc màn hình nhảy theo tiêu đề được) nên
   phải giành lại cỡ chữ của bản thiết kế. */
.stMarkdown h2.vm-h2, [data-testid="stMarkdownContainer"] h2.vm-h2, h2.vm-h2 {
  font-weight: 800; letter-spacing: -0.015em; font-size: 22px;
  margin: 0 0 var(--space-3); padding: 0; color: var(--color-text); }
.stMarkdown h1.vm-h1, [data-testid="stMarkdownContainer"] h1.vm-h1, h1.vm-h1 {
  font-weight: 800; letter-spacing: -0.015em; font-size: 38px;
  line-height: 1.12; margin: 0 0 6px; padding: 0; color: var(--color-text); }
.vm-lede { margin: 0; max-width: 62ch; color: var(--color-neutral-700); font-size: 14.5px; text-wrap: pretty; }
.vm-kicker { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--color-neutral-700); font-weight: 700; }
.vm-label { font-size: 12.5px; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--color-neutral-700); }
.vm-muted { color: var(--color-neutral-700); }
.vm-mono { font-family: var(--font-mono); }
.vm-tag { display: inline-flex; align-items: center; font-size: 11.5px; font-weight: 700;
  padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.vm-tag-accent { background: var(--color-accent-100); color: var(--color-accent-800); }
.vm-tag-ok { background: var(--color-accent-2-700); color: #fff; }  /* 6,46:1 */
.vm-tag-err { background: var(--color-accent-700); color: #fff; }
.vm-tag-neutral { background: var(--color-neutral-200); color: var(--color-neutral-800); }
.vm-num { font-variant-numeric: tabular-nums; }

.vm-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: var(--space-3); }
.vm-tile { background: var(--color-neutral-100); border-radius: var(--radius-lg);
  padding: var(--space-4) var(--space-6); box-shadow: var(--shadow-sm);
  display: flex; flex-direction: column; gap: 3px; }
.vm-tile-k { font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase;
  font-weight: 700; color: var(--color-neutral-700); }
.vm-tile-v { font-weight: 800; letter-spacing: -0.015em; line-height: 1.05;
  font-size: clamp(22px, 2.4vw, 32px); white-space: nowrap; }
.vm-tile-d { font-size: 12px; }

/* Bảng rộng cuộn ngang được, và CHỈ hiện bóng mờ ở rìa khi thật sự còn cột bị cắt:
   hai lớp "local" (màu nền thẻ) trượt theo nội dung và che hai lớp bóng "scroll" khi
   đã cuộn hết — kỹ thuật scroll-shadow bằng CSS thuần, không cần JS. */
.vm-scroll {
  overflow-x: auto;
  scrollbar-width: thin;
  background:
    linear-gradient(to right, var(--color-neutral-100) 40%, transparent) left center / 28px 100% no-repeat local,
    linear-gradient(to left, var(--color-neutral-100) 40%, transparent) right center / 28px 100% no-repeat local,
    radial-gradient(farthest-side at 0 50%,
      color-mix(in srgb, var(--color-neutral-900) 18%, transparent), transparent) left center / 12px 100% no-repeat,
    radial-gradient(farthest-side at 100% 50%,
      color-mix(in srgb, var(--color-neutral-900) 18%, transparent), transparent) right center / 12px 100% no-repeat;
}
.vm-table { width: 100%; border-collapse: collapse; font-size: 13.5px; font-variant-numeric: tabular-nums; }
.vm-table th { text-align: right; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--color-neutral-700); padding: 0 0 var(--space-2) var(--space-4);
  white-space: nowrap; font-weight: 700; }
.vm-table th:first-child { text-align: left; padding-left: 0; }
.vm-table td { text-align: right; padding: 10px 0 10px var(--space-4); }
.vm-table td:first-child { text-align: left; font-weight: 700; padding-left: var(--space-3); }
.vm-table tbody tr { border-top: 1px solid var(--color-divider); }
.vm-row-active td:first-child { border-radius: var(--radius-sm) 0 0 var(--radius-sm); }
.vm-row-active { background: var(--color-accent-100); }

.vm-matrix { width: 100%; border-collapse: separate; border-spacing: 4px;
  font-size: 13.5px; font-variant-numeric: tabular-nums; }
.vm-matrix th { font-size: 11px; color: var(--color-neutral-700); font-weight: 700; text-align: center; }
.vm-matrix th:first-child { text-align: left; }
.vm-matrix th span { font-weight: 400; text-transform: none; letter-spacing: 0; }
.vm-matrix td { text-align: center; padding: 11px 6px; border-radius: var(--radius-sm); font-weight: 700; }
.vm-matrix td:first-child { text-align: left; background: transparent;
  padding-right: var(--space-3); white-space: nowrap; }

.vm-bar { display: flex; height: 22px; border-radius: 999px; overflow: hidden; background: var(--color-neutral-200); }
.vm-taxrow { display: grid; grid-template-columns: 148px minmax(0, 1fr) 64px;
  gap: var(--space-3); align-items: center; margin-bottom: var(--space-3); }
.vm-legend { display: flex; flex-wrap: wrap; gap: var(--space-4); margin-top: var(--space-4);
  font-size: 12px; color: var(--color-neutral-700); }
.vm-swatch { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
  vertical-align: -1px; margin-right: 5px; }

.vm-verdict { border-radius: var(--radius-lg); padding: var(--space-6); box-shadow: var(--shadow-sm); }
.vm-verdict-answer { font-weight: 800; letter-spacing: -0.015em; font-size: 26px;
  line-height: 1.25; text-wrap: pretty; }
.vm-stats { display: flex; flex-wrap: wrap; gap: var(--space-4); margin-top: var(--space-4); font-size: 13px; }
.vm-stat-k { font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.7; }
.vm-stat-v { font-weight: 700; font-variant-numeric: tabular-nums; font-size: 15px; }

.vm-side { display: grid; grid-template-columns: 124px minmax(0, 1fr) auto; gap: var(--space-3);
  align-items: baseline; padding: 9px var(--space-3); border-radius: var(--radius-md);
  margin-bottom: var(--space-2); }
.vm-ctx { font-size: 14.5px; line-height: 1.85; }
.vm-ctx mark { border-radius: 5px; padding: 1px 3px;
  -webkit-box-decoration-break: clone; box-decoration-break: clone; color: var(--color-text); }
.vm-mark-gold { background: var(--color-accent-2-300); }
.vm-mark-pred { background: var(--color-accent-300); }

.vm-err { background: var(--color-neutral-100); border-radius: var(--radius-lg); padding: var(--space-6);
  box-shadow: var(--shadow-sm); margin-bottom: var(--space-3); }
.vm-err-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--space-3);
  margin-bottom: var(--space-3); }
.vm-err-q { font-size: 16px; font-weight: 700; text-wrap: pretty; }
.vm-kv { display: grid; grid-template-columns: 104px minmax(0, 1fr) auto; gap: var(--space-3);
  align-items: baseline; padding: 8px var(--space-3); border-radius: var(--radius-md); margin-bottom: 6px; }

.vm-groupwarn { background: var(--color-accent-100); border-radius: var(--radius-lg);
  padding: var(--space-4) var(--space-6); display: flex; gap: var(--space-3); align-items: flex-start;
  color: var(--color-accent-800); font-size: 13.5px; }

.vm-samples { display: grid; grid-template-columns: 104px minmax(0, 1fr) 190px 88px;
  gap: var(--space-3); align-items: baseline; padding: 10px var(--space-3);
  border-radius: var(--radius-md); background: var(--color-bg); margin-bottom: var(--space-2); }

.vm-cause { background: var(--color-bg); border-radius: var(--radius-md); padding: var(--space-4);
  display: flex; flex-direction: column; gap: 6px; }
.vm-causes { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: var(--space-3); }

.vm-foot { font-size: 11.5px; line-height: 1.7; color: var(--color-neutral-700);
  border-top: 1px solid var(--color-divider); padding-top: var(--space-3); margin-top: var(--space-4); }

/* nhóm thực hiện ở chân thanh bên */
.vm-team { border-top: 1px solid var(--color-divider);
  padding-top: var(--space-3); margin-top: var(--space-4); }
.vm-team .vm-kicker { display: block; margin-bottom: var(--space-2); }
.vm-team-row { display: flex; align-items: baseline; justify-content: space-between;
  gap: var(--space-2); font-size: 12px; line-height: 1.75; }
.vm-team-row .vm-mono { font-size: 11px; }
.vm-foot { border-top: 0; margin-top: var(--space-2); padding-top: 0; }

@media (max-width: 900px) {
  .stMainBlockContainer, [data-testid="stMainBlockContainer"],
  [data-testid="stAppViewBlockContainer"] { padding: var(--space-4); }
  .vm-h1 { font-size: 30px; }
  .vm-taxrow, .vm-samples, .vm-kv, .vm-side { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""


def css() -> str:
    return "<style>" + CSS.replace("__TOKENS__", TOKENS) + "</style>"


# ── nền ô nhiệt cho ma trận E1–E5 (thang của bản thiết kế) ───────────────────
def heat(v: float | None) -> tuple[str, str]:
    if v is None:
        return ("var(--color-neutral-200)", "var(--color-neutral-700)")
    if v >= 85:
        return ("#aebf92", "#272e1b")
    if v >= 60:
        return ("#ccdbb2", "#272e1b")
    if v >= 40:
        return ("#e1eecc", "#3d472b")
    if v >= 20:
        return ("#ffe1d0", "#643312")
    return ("#ffc6a5", "#643312")
