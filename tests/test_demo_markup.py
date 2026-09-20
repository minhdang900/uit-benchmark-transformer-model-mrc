"""Ngữ nghĩa của phần HTML do ViMRC Console tự sinh (demo/console/markup.py).

Streamlit tự sinh phần lớn DOM, nhưng các khối nội dung là HTML do đồ án viết —
nên ngữ nghĩa trợ năng (tiêu đề thật, tên gọi cho biểu đồ) phải kiểm thử được ở
đây, nơi không cần chạy trình duyệt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

from console.markup import bar, heading, logo, team  # noqa: E402
from console.theme import css  # noqa: E402

MEMBERS = [("Nguyễn Quang Lâm", "25210289"), ("Vỏ Cẩm Thu", "25210342")]

SEGS = [
    {"label": "trả lời sai", "n": 544, "pct": 34.2, "bg": "#c67139"},
    {"label": "từ chối sai", "n": 335, "pct": 21.1, "bg": "#728157"},
]


def test_heading_renders_a_real_h2_element():
    assert heading("Các lần huấn luyện").startswith("<h2")


def test_heading_keeps_the_presentation_class_so_styling_is_unchanged():
    assert "vm-h2" in heading("Các lần huấn luyện")


def test_heading_carries_an_inline_style_when_one_is_given():
    assert "margin:0" in heading("Ranh giới từ ghép", style="margin:0")


def test_heading_omits_the_style_attribute_when_none_is_given():
    assert "style=" not in heading("Các lần huấn luyện")


# ── biểu đồ thanh: dựng bằng div nên phải tự khai báo tên gọi ────────────────

def test_bar_declares_itself_as_a_graphic():
    assert "role='img'" in bar("PhoBERT s42", SEGS, total=1590)


def test_bar_names_the_row_it_describes():
    assert "PhoBERT s42" in _aria(bar("PhoBERT s42", SEGS, total=1590))


def test_bar_states_every_segment_and_its_count_in_the_accessible_name():
    label = _aria(bar("PhoBERT s42", SEGS, total=1590))
    assert "trả lời sai 544" in label
    assert "từ chối sai 335" in label


def test_bar_states_the_total_in_the_accessible_name():
    assert "1590" in _aria(bar("PhoBERT s42", SEGS, total=1590))


def test_bar_still_renders_one_coloured_segment_per_entry():
    assert bar("PhoBERT s42", SEGS, total=1590).count("<div") == len(SEGS) + 1


# ── logo chữ "ViMRC" ────────────────────────────────────────────────────────

def test_logo_is_an_svg():
    assert logo().lstrip().startswith("<svg")


def test_logo_names_the_app_for_assistive_technology():
    assert "<title>ViMRC</title>" in logo()


def test_logo_needs_no_font_at_render_time():
    # Streamlit nhúng logo qua <img>, nơi phông của trang không áp được — nên
    # chữ phải là đường vẽ sẵn, không phải phần tử <text>.
    markup = logo()
    assert "<text" not in markup
    assert "font-family" not in markup


def test_logo_uses_the_ink_token_so_it_matches_the_design():
    assert "#201e1d" in logo().lower()


def test_logo_declares_a_viewbox_so_it_scales():
    assert "viewBox=" in logo()


# ── khối nhóm thực hiện ở thanh bên ─────────────────────────────────────────

def test_team_lists_every_member():
    markup = team(MEMBERS)
    assert "Nguyễn Quang Lâm" in markup
    assert "Vỏ Cẩm Thu" in markup


def test_team_shows_each_student_id():
    markup = team(MEMBERS)
    assert "25210289" in markup
    assert "25210342" in markup


def test_team_is_labelled_so_the_names_are_not_bare_text():
    assert "Nhóm thực hiện" in team(MEMBERS)


def test_team_escapes_member_names():
    assert "<script>" not in team([("<script>", "1")])


# ── thanh bên: chỗ trống Streamlit dành sẵn cho logo ────────────────────────

def test_stylesheet_removes_the_logo_spacer_streamlit_reserves():
    assert "stLogoSpacer" in css()


def test_stylesheet_collapses_the_empty_sidebar_header():
    assert "stSidebarHeader" in css()


def test_stylesheet_sizes_the_sidebar_logo():
    assert "stSidebarLogo" in css()


def test_stylesheet_provides_a_visually_hidden_class():
    # Streamlit đặt alt="Logo" cho ảnh logo và không cho đổi; tên ứng dụng phải
    # có ở dạng chữ đọc được cho trình đọc màn hình.
    assert ".vm-sr-only" in css()


def test_stylesheet_targets_the_main_block_container_streamlit_actually_renders():
    # Streamlit đặt lớp `stMainBlockContainer` cho vùng nội dung chính; chỉ nhắm
    # vào tên cũ thì khoảng đệm của bản thiết kế không áp được.
    assert "stMainBlockContainer" in css()


def _aria(markup: str) -> str:
    start = markup.index("aria-label='") + len("aria-label='")
    return markup[start:markup.index("'", start)]
