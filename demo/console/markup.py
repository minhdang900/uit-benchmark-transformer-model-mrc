"""Các khối HTML do đồ án tự sinh, tách khỏi Streamlit để kiểm thử được.

Streamlit dựng phần lớn DOM, nhưng nội dung từng trang là HTML viết tay. Ngữ
nghĩa trợ năng nằm ở đây: tiêu đề mục là ``<h2>`` thật (đọc màn hình nhảy theo
tiêu đề được), và biểu đồ dựng bằng ``<div>`` phải có tên gọi để không chỉ tồn
tại dưới dạng chiều rộng.
"""
from __future__ import annotations

import html


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def heading(text: str, style: str | None = None) -> str:
    """Tiêu đề mục: phần tử tiêu đề thật, giữ nguyên lớp trình bày ``vm-h2``."""
    attr = f" style='{esc(style)}'" if style else ""
    return f"<h2 class='vm-h2'{attr}>{esc(text)}</h2>"


def team(members: list[tuple[str, str]]) -> str:
    """Khối nhóm thực hiện ở chân thanh bên: họ tên và MSSV."""
    rows = "".join(
        f"<div class='vm-team-row'><span>{esc(name)}</span>"
        f"<span class='vm-mono vm-muted'>{esc(mssv)}</span></div>"
        for name, mssv in members)
    return (f"<div class='vm-team'><div class='vm-kicker'>Nhóm thực hiện</div>{rows}</div>")


def bar(label: str, segments: list[dict], total) -> str:
    """Thanh chia đoạn, kèm tên gọi đọc được: dữ liệu không chỉ nằm ở chiều rộng."""
    parts = ", ".join(f"{s['label']} {s['n']}" for s in segments)
    name = f"{label}: {total} lỗi — {parts}"
    segs = "".join(
        f"<div title='{esc(s['label'])}: {esc(s['n'])}' style='height:100%;"
        f"width:{s['pct']:.2f}%;background:{esc(s['bg'])}'></div>" for s in segments)
    return f"<div class='vm-bar' role='img' aria-label='{esc(name)}'>{segs}</div>"

# Chữ "ViMRC" dựng sẵn thành đường vẽ từ Be Vietnam Pro ExtraBold (cùng phông
# với tiêu đề). Streamlit nhúng logo qua <img>, nơi phông của trang KHÔNG áp
# được — nên không dùng <text>, và logo không phụ thuộc máy có cài phông hay không.
_LOGO_VIEWBOX = "0 0 3422 793"
_LOGO_PATHS = (
    '<path d="M249 783 0 43H181L365 619L549 43H730L481 783Z"/><path d="M897 190Q854 190 826.0 162.0Q798 134 798 96Q798 56 826.0 28.0Q854 0 897 0Q940 0 968.0 28.0Q996 56 996 96Q996 134 968.0 162.0Q940 190 897 190ZM811 783V253H982V783Z"/><path d="M1114 783V43H1348L1501 403L1660 43H1886V783H1715V310L1580 615H1420L1285 308V783Z"/><path d="M2014 783V43H2327Q2409 43 2470.0 71.0Q2531 99 2564.0 151.0Q2597 203 2597 275Q2597 349 2561.5 402.0Q2526 455 2462 482L2615 783H2424L2287 506H2187V783ZM2325 192H2187V367H2325Q2370 367 2397.0 343.0Q2424 319 2424 280Q2424 240 2397.0 216.0Q2370 192 2325 192Z"/><path d="M3083 793Q2991 793 2921.0 760.5Q2851 728 2804.5 673.5Q2758 619 2734.5 551.0Q2711 483 2711 412Q2711 342 2734.5 274.0Q2758 206 2804.5 152.0Q2851 98 2921.0 65.5Q2991 33 3083 33Q3162 33 3217.5 52.5Q3273 72 3310.5 103.5Q3348 135 3370.5 169.5Q3393 204 3404.0 235.0Q3415 266 3418.5 286.0Q3422 306 3422 306H3251Q3251 306 3245.5 288.0Q3240 270 3223.5 246.5Q3207 223 3174.0 205.0Q3141 187 3085 187Q3020 187 2976.5 220.0Q2933 253 2911.5 304.5Q2890 356 2890 412Q2890 468 2911.5 520.0Q2933 572 2976.5 605.0Q3020 638 3085 638Q3141 638 3174.0 620.0Q3207 602 3223.5 579.0Q3240 556 3245.5 538.0Q3251 520 3251 520H3422Q3422 520 3418.5 539.5Q3415 559 3404.0 590.5Q3393 622 3370.5 656.5Q3348 691 3310.5 722.5Q3273 754 3217.5 773.5Q3162 793 3083 793Z"/>'
)


def logo(color: str = "#201e1d") -> str:
    """Logo chữ ViMRC dạng SVG, màu mặc định là token --color-text."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{_LOGO_VIEWBOX}' "
            f"role='img' aria-label='ViMRC'><title>ViMRC</title>"
            f"<g fill='{esc(color)}'>{_LOGO_PATHS}</g></svg>")
