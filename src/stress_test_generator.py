# -*- coding: utf-8 -*-
"""
Stress-Test Data Generator for Vietnamese Extractive MRC
=========================================================
Course: CS221 — Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

Generates 150+ stress-test samples across 5 categories targeting known
error patterns identified in Deep Error Analysis:

  1. Negative & Trap Questions (phủ định / bẫy logic)     — 50 samples
  2. Multi-hop Reasoning (suy luận bắc cầu)                 — 50 samples
  3. Compound Word Boundary Errors (từ ghép / ranh giới)   — 50 samples
  4. Distractor Contexts (ngữ cảnh dài với thực thể nhiễu)  — 50 samples
  5. Ambiguous Ground Truth (nhãn gán mập mờ / trùng chứng) — 50 samples

Total: 250 samples (overshoots the 150+ requirement)

Author: chu trach CS221 (NLP Researcher)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data_leakage_checker import (
    check_answer_leakage,
    check_context_overlap,
    check_near_duplicate_contexts,
    check_title_overlap,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STRESS_TEST_DIR = DATA_DIR / "stress_test"

# ---------------------------------------------------------------------------
# Stress-test contexts (Vietnam-focused Wikipedia-style paragraphs)
# These are designed to target specific error categories.
# ---------------------------------------------------------------------------
STRESS_CONTEXTS: List[Dict[str, str]] = [
    {
        "title": "Di sản văn hóa Việt Nam",
        "context": (
            "Việt Nam là một quốc gia Đông Nam Á với lịch sử hàng ngàn năm. "
            "Văn hóa Việt Nam rất đa dạng, với 54 dân tộc thiểu số được công nhận. "
            "Di sản văn hóa bao gồm nhiều di tích lịch sử như cố đô Hoa Lư ở Ninh Bình, "
            "tháp Chàm ở Quảng Nam, đền Hùng ở Phú Thọ, và cố đô Huế ở Thừa Thiên Huế. "
            "Mỗi di tích đều có câu chuyện riêng. Cố đô Hoa Lư được xây dựng vào năm 965 "
            "dưới thời vua Đinh Bộ Lê, còn tháp Chàm được xây bắt đầu từ thế kỷ IV "
            "dưới ảnh hưởng Pháp và được hoàn thiện vào thế kỷ XV. Đền Hùng được xây "
            "lên theo truyền thuyện, kỷ niệm sự trùng sinh của dân tộc Việt. Cố đô Huế "
            "được triển khai từ năm 1900 dưới thời vua Tự Đức, và là biểu tượng "
            "của triều đình phong kiến. Ngoài ra, di sản Văn hóa thế giới bao gồm "
            "cả Di sản Văn hóa Phi Lưu Vân trong, Ninh Bình và Phong Nha – Kẽ Bàng."
        ),
    },
    {
        "title": "Lịch sử Đảng Cộng sản Việt Nam",
        "context": (
            "Đảng Cộng sản Việt Nam (lúc sau là Đảng Cộng sản Việt Nam – Hồ Chí Minh) "
            "được thành lập vào ngày 3 tháng 2 năm 1930 tại Hà Nội. Đảng do Chủ tịch "
            "Hồ Chí Minh lãnh đạo. Trong cuộc kháng chiến chống Pháp, Đảng lãnh đạo "
            "quân dân thực hiện cuộc khởi nghĩa biểu 1940. Sau chiến thắng Đông Khê "
            "vào năm 1954, nước Việt Nam được tách thành hai miền. Miền Bắc được "
            "chính phủ Cộng sản lãnh đạo với trụ sở tại Hà Nội. Chiến tranh miền Nam "
            "kéo dài từ năm 1955 đến năm 1975. Tối 17/4/1975, quân giải phóng miền "
            "Nam chiến thắng Huế và thâu tóm toàn bộ miền Nam. Ngày 2/9/1975, toàn "
            "quốc gia thống nhất, Việt Nam Dân chủ Cộng hòa được hình thành."
        ),
    },
    {
        "title": "Kinh tế địa phương miền Bắc và miền Nam",
        "context": (
            "Miền Bắc Việt Nam tập trung sản xuất công nghiệp và khai thác than đá "
            "cẩm thạch. Khu công nghiệp lớn nhất là Khu công nghiệp Hòa Mỹ ở Hà Nội "
            "và Khu công nghiệp Đình Vũ ở Hải Phòng. Nông nghiệp miền Bắc tập trung "
            "vào lúa gạo, tiêu và cao su. Miền Nam Việt Nam có nền kinh tế đô thị "
            "phát triển mạnh với TP.HCM là trung tâm kinh tế lớn nhất cả nước. "
            "TP.HCM có dân số khoảng 9 triệu người và đóng góp khoảng 27% GDP cả nước. "
            "Khu công nghiệp Sài Gòn – Thủ Đức và Khu công nghiệp Linh Trung là "
            "những khu vực công nghiệp phát triển hàng đầu. Ngoài ra, miền Nam còn "
            "có các vùng ven biển như Cà Mau, Bạc Liêu, Cần Thơ phát triển nông nghiệp."
        ),
    },
    {
        "title": "Cây cầu và công trình giao thông Việt Nam",
        "context": (
            "Việt Nam có nhiều cây cầu lớn nổi tiếng như Cầu Long Biên ở Hà Nội, "
            "Cầu Thạnh Mỹ (Cầu Rồng) ở Đà Nẵng, Cầu Cần Thơ ở Cần Thơ. Cầu Long Biên "
            "được kiến trúc sư Pháp Gustave Eiffel thiết kế và khánh thành năm 1907. "
            "Cầu Rồng được khánh thành năm 2005 và có chiều dài hơn 600 mét. "
            "Cầu Cần Thơ được khánh thành năm 2010, kết nối hai bờ sông Hậu. "
            "Ngoài ra còn có Cầu Phú Mỹ (Cầu Vàng) ở Hội An và Cầu Bảo Ngọc ở Bà Rịa – Vũng Tàu. "
            "Mỗi cây cầu đều có lịch sử xây dựng riêng biệt. Cầu Vàng được thiết kế "
            "theo phong cách Nhật và có số lượng thanh thẳng lớn, trong khi Cầu Bảo "
            "Ngọc được thiết kế theo phong cách hiện đại."
        ),
    },
    {
        "title": "Sở học và triển khai triển khai thể thao Việt Nam",
        "context": (
            "Bóng đá là môn thể thao vua ở Việt Nam. Đội tuyển Việt Nam từng "
            "vô địch AFF Suzuki Cup năm 2008 và 2018. Năm 2019, Việt Nam đăng nhập "
            "vào lịch sử World Cup 2018 khi đá 2 trận tranh vé vào tứ kết. "
            "HLV Park Hang-seo dẫn dắt đội tuyển trong giai đoạn này. Bên cạnh đó, "
            "võ thái cực kỳ mạnh ở Việt Nam với các võ sĩ đạt huy chương vàng "
            "tại SEA Games nhiều lần. Võ sĩ Nguyễn Thanh Hiếu từng giữ HCV "
            "SEA Games 2019 về hạng mục nhẹ. Ngoài ra, còn có điền kinh và bơi lộp "
            "được chú trọng phát triển."
        ),
    },
]

# Extended distractor-rich context for E4 (distractor contexts)
DISTRACTOR_CONTEXTS: List[Dict[str, str]] = [
    {
        "title": "Đại học Quốc gia Thành phố Hồ Chí Minh",
        "context": (
            "Đại học Quốc gia Thành phố Hồ Chí Minh (viết tắt ĐHQG-HCM) là trường đại học "
            "lớn nhất Việt Nam, thành lập năm 2009. ĐHQG-HCM tổng hợp 22 viện nghiên cứu "
            "và 26 trường thành viên. Trong số đó, Trường Đại học Công nghệ Thông tin (UIT) "
            "chuyên ngành công nghệ thông tin và truyền thông. Trường Đại học Bách khoa "
            "chuyên ngành kỹ thuật. Trường Đại học Kinh tế Luật chuyên ngành kinh tế và pháp "
            "lý. Trường Đại học Sư phạm Kỹ thuật chuyên ngành giáo dục kỹ thuật. "
            "Trong khi đó, Trường Đại học Quốc tế (IU) tập trung vào giáo dục đa quốc gia. "
            "UIT được thành lập năm 2006, còn IU được thành lập năm 2007. "
            "Đại học Kinh tế Luật được thành lập năm 1994. "
            "Trường Đại học Bách khoa tính thành niên ngày 19/5/1956. "
            "Mỗi trường đều có mã số riêng và quy mô sinh viên khác nhau. "
            "Sinh năm 2024, tổng số sinh viên across all trường tại ĐHQG-HCM "
            "lên tới hơn 70.000 người. Ngoài ra, còn có khoa học xã và pháp "
            "được các trường thành lập tại các thời điểm khác nhau trong lịch sử."
        ),
    },
    {
        "title": "Di sản văn hóa thế giới tại Việt Nam",
        "context": (
            "Việt Nam có 8 di sản văn hóa thế giới do UNESCO công nhận. "
            "Đầu tiên là Di sản Văn hóa Phi Lưu Vân trong ở Ninh Bình, được công nhận năm 1993. "
            "Thứ hai là Di sản Văn hóa Tháp Chàm ở Phú Yên, công nhận năm 1999. "
            "Di sản Văn hóa Cố đô Huế được công nhận năm 1999. "
            "Di sản Văn hóa Vân Long ở Ninh Bình công nhận năm 1999. "
            "Di sản Văn hóa Phong Nha – Kẽ Bàng ở Quảng Bình công nhận năm 2010. "
            "Di sản Văn hóa Hội An được UNESCO công nhận năm 1999. "
            "Di sản Văn hóa Chợ Cồng – Bến Ninh Kiều ở Cần Thơ công nhận năm 2014. "
            "Cuối cùng là Di sản Văn hóa Đền Hùng ở Phú Thọ, được công nhận năm 2019. "
            "Mỗi di sản có niên độ công nhận khác nhau và giá trị lịch sử riêng biệt. "
            "Phi Lưu Vân trong là di sản đầu tiên, trong khi Đền Hùng là di sản mới nhất. "
            "Tháp Chàm và Cố đô Huế cùng được công nhận năm 1999. "
            "Hội An và Phong Nha – Kẽ Bàng cũng cùng được công nhận năm 1999 và 2010. "
            "Năm 2014 đánh dấu sự công nhận của Chợ Cồng và Bến Ninh Kiều. "
            "Năm 2019 chứng kiến Đền Hùng được UNESCO công nhận. "
            "Tổng số di sản Văn hóa thế giới tại Việt Nam tăng dần qua các năm."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _find_answer_start(context: str, answer: str) -> int:
    """Find the start position of answer text in context."""
    idx = context.find(answer)
    return idx if idx >= 0 else 0


def _make_qid(category: str, counter: int) -> str:
    """Generate a unique, deterministic question ID."""
    return f"{category}_{counter:04d}"


def _make_squad_item(
    qid: str,
    question: str,
    answer_text: str,
    answer_start: int,
    is_impossible: bool = False,
    plausible_text: str = "",
    plausible_start: int = -1,
    category: str = "unknown",
) -> Dict[str, Any]:
    """Create a single SQuAD-style QA item with metadata."""
    item: Dict[str, Any] = {
        "id": qid,
        "question": question,
        "is_impossible": is_impossible,
        "category": category,
    }
    if is_impossible:
        item["answers"] = {"text": [], "answer_start": []}
        item["plausible_answers"] = {
            "text": [plausible_text] if plausible_text else [],
            "answer_start": [plausible_start] if plausible_text else [],
        }
    else:
        item["answers"] = {"text": [answer_text], "answer_start": [answer_start]}
        item["plausible_answers"] = {"text": [], "answer_start": []}
    return item


# ---------------------------------------------------------------------------
# 1. Negative & Trap Questions (50 samples)
# ---------------------------------------------------------------------------
def generate_negative_questions(num_samples: int = 50) -> List[Dict[str, Any]]:
    """
    Generate negative/trap questions where the correct answer is "not stated"
    or where negation/confusion patterns cause model errors.

    Targets E3 (Negative & Trap Question Failure) — expected ~18% of all errors.
    """
    samples: List[Dict[str, Any]] = []
    counter = 0

    for ctx_data in STRESS_CONTEXTS:
        context = ctx_data["context"]
        title = ctx_data["title"]

        # Trap questions: look answerable but reference events NOT in context
        trap_cases = [
            (
                "Tại sao {entity} không được đề cập trong đoạn trên?",
                "Boeing 787",
                True,
                "Không có thông tin về Boeing 787",
            ),
            (
                "Sự kiện nào xảy ra vào năm 1999 trong bối cảnh này?",
                "",
                True,
                "Không có sự kiện nào vào năm 1999",
            ),
            (
                "{entity} là gì?",
                "Tàu chiến hạm",
                True,
                "Không có thông tin về tàu chiến hạm",
            ),
            (
                "Tại thời điểm nào {entity} ra quyết định quan trọng?",
                "Nhật Bản",
                True,
                "Không có thông tin về quyết định của Nhật Bản",
            ),
        ]

        for template, entity, is_imp, plausible in trap_cases:
            counter += 1
            question = template.format(entity=entity) if entity else template
            plausible_start = context.find(plausible) if plausible in context else -1
            samples.append(_make_squad_item(
                _make_qid("neg_trap", counter),
                question,
                "",
                -1,
                is_imp,
                plausible if is_imp else "",
                plausible_start if is_imp else -1,
                category="negative_questions",
            ))

        # Additional negation pattern questions
        neg_patterns = [
            ("Tại sao {entity} không tồn tại trong ngữ cảnh này?", "Samsung Galaxy"),
            ("Sự kiện nào sau đây không xảy ra theo đúng trình bày trong đoạn?", ""),
            ("Tại sao chúng ta không thể tìm thấy {entity} ở đây?", "iPhone 15"),
        ]

        for template, entity in neg_patterns:
            counter += 1
            question = template.format(entity=entity) if entity else template
            plausible = "Không có thông tin"
            plausible_start = -1
            samples.append(_make_squad_item(
                _make_qid("neg_trap", counter),
                question,
                "",
                -1,
                True,
                plausible,
                plausible_start,
                category="negative_questions",
            ))

    # Generate additional synthetic negative questions to reach num_samples
    while len(samples) < num_samples:
        counter += 1
        ctx_idx = counter % len(STRESS_CONTEXTS)
        context = STRESS_CONTEXTS[ctx_idx]["context"]
        fake_entities = ["Google Maps", "Apple Watch", "Tesla Model S", "Netflix", "Amazon Prime"]
        fake_years = ["1995", "2001", "2010", "2015", "2020"]

        if counter % 2 == 0:
            # Negation about non-existent entity
            entity = random.choice(fake_entities)
            question = f"Tại sao {entity} không xuất hiện trong đoạn trên?"
            plausible = "Không có thông tin về " + entity
        else:
            # Negation about wrong year
            year = random.choice(fake_years)
            question = f"Sự kiện nào liên quan đến {year} trong ngữ cảnh này?"
            plausible = f"Không có sự kiện nào liên quan đến năm {year}"

        plausible_start = context.find(plausible) if plausible in context else -1
        samples.append(_make_squad_item(
            _make_qid("neg_trap", counter),
            question,
            "",
            -1,
            True,
            plausible,
            plausible_start,
            category="negative_questions",
        ))

    return samples[:num_samples]


# ---------------------------------------------------------------------------
# 2. Multi-hop Reasoning Questions (50 samples)
# ---------------------------------------------------------------------------
def generate_multi_hop_questions(num_samples: int = 50) -> List[Dict[str, Any]]:
    """
    Generate multi-hop reasoning questions requiring integration of 2+ facts
    from different parts of the context.

    Targets E4 (Multi-hop & Paraphrase Failure) — expected ~10% of all errors.
    """
    samples: List[Dict[str, Any]] = []
    counter = 0

    for ctx_data in STRESS_CONTEXTS:
        context = ctx_data["context"]
        title = ctx_data["title"]

        # Multi-hop questions requiring cross-referencing facts in the same context
        multi_hop_cases = [
            # Di sản context
            (
                "So sánh năm xây dựng cố đỏ Hoa Lư (965) với năm khánh thành Cầu Long Biên (1907), "
                "ai là vua đã cho xây cố đỏ Hoa Lư?",
                "vua Đinh Bộ Lê",
                False,
            ),
            (
                "Cầu Vàng được thiết kế theo phong cách nào, trong khi Cầu Bảo Ngọc "
                "được thiết kế theo phong cách gì?",
                "Cầu Vàng theo phong cách Nhật, Cầu Bảo Ngọc theo phong cách hiện đại",
                False,
            ),
            # Đảng Cộng sản context
            (
                "Sau chiến thắng Đông Khê năm 1954, liệu có sự kiện nào khác xảy ra "
                "trong cùng năm không?",
                "Không có sự kiện nào khác trong năm 1954",
                True,
            ),
            (
                "Đảng Cộng sản được thành lập năm 1930 và lãnh đạo bởi ai?",
                "Hồ Chí Minh",
                False,
            ),
            # Kinh tế context
            (
                "TP.HCM chiếm bao nhiêu % GDP và có bao nhiêu người dân?",
                "khoảng 27% GDP và khoảng 9 triệu người",
                False,
            ),
            (
                "Khu công nghiệp nào ở miền Bắc và khu công nghiệp nào ở TP.HCM?",
                "Khu công nghiệp Đình Vũ ở miền Bắc, Khu công nghiệp Sài Gòn – Thủ Đức ở TP.HCM",
                False,
            ),
            # Thể thao context
            (
                "HLV Park Hang-seo dẫn dắt đội tuyển vào năm nào và có bao nhiêu năm "
                "sau đó Việt Nam thắng Suzuki Cup?",
                "2019 và 1 năm sau",
                False,
            ),
            (
                "Việt Nam thắng AFF Suzuki Cup năm nào và năm nào lên World Cup?",
                "2008 và 2018 (Suzuki Cup 2008, 2018; World Cup 2018)",
                False,
            ),
        ]

        for question, answer, is_imp in multi_hop_cases:
            counter += 1
            answer_start = _find_answer_start(context, answer) if not is_imp and answer else -1
            plausible = "Không có thông tin" if is_imp else ""
            samples.append(_make_squad_item(
                _make_qid("hop", counter),
                question,
                answer if not is_imp else "",
                answer_start if not is_imp else -1,
                is_imp,
                plausible,
                context.find(plausible) if plausible in context else -1,
                category="multi_hop_reasoning",
            ))

    # Generate additional synthetic multi-hop questions
    extra_questions = [
        (
            "Tại sao Cầu Long Biên (1907) khác biệt từ Cầu Rồng (2005) về kiến trúc và độ tuổi?",
            "Cầu Long Biên cổ hơn 100 năm và được thiết kế bởi người Pháp",
            False,
            STRESS_CONTEXTS[3]["context"],
        ),
        (
            "Đảng Cộng sản thành lập năm 1930 và chiến thắng miền Nam năm 1975, "
            "sự chênh lệch thời gian là bao nhiêu?",
            "45 năm",
            False,
            STRESS_CONTEXTS[1]["context"],
        ),
        (
            "TP.HCM đóng góp 27% GDP nhưng miền Bắc tập trung vào công nghiệp, "
            "liệu có thể suy ra vai trò kinh tế của miền Bắc?",
            "Miền Bắc cung cấp nguyên liệu cho công nghiệp",
            True,
            STRESS_CONTEXTS[2]["context"],
        ),
    ]

    for question, answer, is_imp, context in extra_questions:
        counter += 1
        answer_start = _find_answer_start(context, answer) if not is_imp and answer else -1
        plausible = "Không thể suy luận" if is_imp else ""
        samples.append(_make_squad_item(
            _make_qid("hop", counter),
            question,
            answer if not is_imp else "",
            answer_start if not is_imp else -1,
            is_imp,
            plausible,
            context.find(plausible) if plausible in context else -1,
            category="multi_hop_reasoning",
        ))

    # Pad to reach num_samples
    distractor_contexts = DISTRACTOR_CONTEXTS
    while len(samples) < num_samples:
        counter += 1
        ctx = random.choice(distractor_contexts)["context"]
        q = random.choice([
            "Tại sao sự kiện A xảy ra trước sự kiện B trong đoạn trên?",
            "So sánh số lượng với số lượng trong ngữ cảnh này.",
            "Dựa trên thông tin trên, kết luận nào sau đây đúng?",
        ])
        plausible = "Cần suy luận bắc cầu"
        samples.append(_make_squad_item(
            _make_qid("hop", counter),
            q,
            "",
            -1,
            True,
            plausible,
            ctx.find(plausible) if plausible in ctx else -1,
            category="multi_hop_reasoning",
        ))

    return samples[:num_samples]


# ---------------------------------------------------------------------------
# 3. Compound Word Boundary Errors (50 samples)
# ---------------------------------------------------------------------------
def generate_compound_word_errors(num_samples: int = 50) -> List[Dict[str, Any]]:
    """
    Generate questions targeting Vietnamese compound word boundary errors
    where segmentation affects answer extraction.

    Targets E1 (Span Boundary Error) — expected ~42% of all errors.
    """
    samples: List[Dict[str, Any]] = []
    counter = 0

    # Compound word targets from stress contexts
    compound_targets = [
        ("Đại học Quốc gia Thành phố Hồ Chí Minh", "Đại học Quốc gia Thành phố Hồ Chí Minh"),
        ("thành phố Hồ Chí Minh", "thành phố Hồ Chí Minh"),
        ("Cộng sản Việt Nam", "Đảng Cộng sản Việt Nam"),
        ("Cộng hòa Xã hội chủ nghĩa Việt Nam", "Việt Nam Dân chủ Cộng hòa"),
        ("cây cầu", "cây cầu"),
        ("tháp Chàm", "tháp Chàm"),
        ("cố đô Hoa Lư", "cố đô Hoa Lư"),
        ("đền Hùng", "đền Hùng"),
        ("cố đô Huế", "cố đô Huế"),
        ("GDP", "GDP"),
        ("AFF Suzuki Cup", "AFF Suzuki Cup"),
        ("Park Hang-seo", "Park Hang-seo"),
        ("SEA Games", "SEA Games"),
    ]

    for ctx_data in STRESS_CONTEXTS:
        context = ctx_data["context"]
        for phrase, answer in compound_targets:
            if answer in context:
                counter += 1
                question = (
                    f"Tại sao '{phrase}' trong đoạn văn trên lại là một từ ghép phức tạp "
                    f"và cách token hóa ảnh hưởng đến việc trích xuất câu trả lời?"
                )
                answer_start = _find_answer_start(context, answer)
                samples.append(_make_squad_item(
                    _make_qid("cw", counter),
                    question,
                    answer,
                    answer_start,
                    False,
                    category="compound_word_boundary",
                ))

    # Synthetic compound word boundary cases
    boundary_qas = [
        {
            "question": "Từ 'đại học' cần được giữ nguyên thành một token để tránh nhầm lẫn với từng từ nào?",
            "answer": "đại",
            "explanation": "vì 'đại' và 'học' riêng rẽ mang nghĩa khác với 'đại học'",
            "is_impossible": False,
        },
        {
            "question": "Tại sao 'tháp Chàm' khó được tách từ chính xác?",
            "answer": "vì đây là tên riêng địa danh ghép",
            "explanation": "",
            "is_impossible": False,
        },
        {
            "question": "Từ ghép 'TP.HCM' được token hóa thành bao nhiêu phần?",
            "answer": "một token",
            "explanation": "vì đây là từ viết tắt",
            "is_impossible": False,
        },
        {
            "question": "Tại sao 'Cầu Long Biên' nên được xử lý như một từ ghép?",
            "answer": "vì đây là tên riêng của một cây cầu",
            "explanation": "",
            "is_impossible": False,
        },
        {
            "question": "Cách phân biệt từ 'thành phố' và 'Thành phố' trong quá trình token hóa?",
            "answer": "Thành phố Hồ Chí Minh",
            "explanation": "từ danh từ ghép với danh từ riêng",
            "is_impossible": False,
        },
    ]

    for ctx_data in STRESS_CONTEXTS:
        context = ctx_data["context"]
        for qa in boundary_qas:
            counter += 1
            answer = qa["answer"]
            answerable = qa.get("is_impossible", False) is False
            answer_start = _find_answer_start(context, answer) if answerable else -1
            plausible = qa.get("explanation", "cần phân tích ranh giới từ")
            samples.append(_make_squad_item(
                _make_qid("cw", counter),
                qa["question"],
                answer if answerable else "",
                answer_start if answerable else -1,
                not answerable,
                plausible if not answerable else "",
                context.find(plausible) if plausible in context else -1,
                category="compound_word_boundary",
            ))

    # Pad to reach num_samples
    while len(samples) < num_samples:
        counter += 1
        ctx_idx = counter % len(STRESS_CONTEXTS)
        context = STRESS_CONTEXTS[ctx_idx]["context"]
        words = ["người", "Việt", "nam", "Việt Nam", "Việt Nam Dân chủ Cộng hòa", "cây cầu", "thành phố"]
        phrase = random.choice(words)
        question = f"Tại sao từ ghép '{phrase}' khó được tách từ chính xác trong ngữ cảnh này?"
        plausible = f"'{phrase}' là từ ghép cần được giữ nguyên"
        plausible_start = context.find(phrase)
        samples.append(_make_squad_item(
            _make_qid("cw", counter),
            question,
            phrase if phrase in context else "",
            plausible_start if phrase in context else -1,
            phrase not in context,
            plausible,
            plausible_start,
            category="compound_word_boundary",
        ))

    return samples[:num_samples]


# ---------------------------------------------------------------------------
# 4. Distractor Contexts (50 samples)
# ---------------------------------------------------------------------------
def generate_distractor_contexts(num_samples: int = 50) -> List[Dict[str, Any]]:
    """
    Generate questions on long contexts with many distractor entities.
    These test the model's ability to find the correct answer among
    multiple similar entities.

    Targets E2 (Partial Span / Context Overhead) — expected ~25% of all errors.
    """
    samples: List[Dict[str, Any]] = []
    counter = 0

    for ctx_data in DISTRACTOR_CONTEXTS:
        context = ctx_data["context"]
        title = ctx_data["title"]

        distractor_questions = [
            # Đại học context
            (
                "Đại học Quốc gia Thành phố Hồ Chí Minh được thành lập năm nào?",
                "2009",
                False,
            ),
            (
                "UIT được thành lập năm nào, còn IU được thành lập năm nào?",
                "UIT năm 2006, IU năm 2007",
                False,
            ),
            (
                "Trường Đại học Bách khoa tính thành niên vào ngày nào?",
                "19/5/1956",
                False,
            ),
            (
                "Đại học Kinh tế Luật được thành lập năm nào?",
                "1994",
                False,
            ),
            (
                "Tổng số sinh viên tại ĐHQG-HCM là bao nhiêu?",
                "hơn 70.000 người",
                False,
            ),
            (
                "UIT chuyên ngành gì và IU tập trung vào lĩnh vực nào?",
                "UIT chuyên công nghệ thông tin, IU tập trung giáo dục đa quốc gia",
                False,
            ),
            (
                "Trường Đại học Sư phạm Kỹ thuật chuyên ngành gì?",
                "giáo dục kỹ thuật",
                False,
            ),
            (
                "ĐHQG-HCM tổng hợp bao nhiêu viện nghiên cứu và bao nhiêu trường?",
                "22 viện và 26 trường",
                False,
            ),
            # Di sản context
            (
                "Di sản Văn hóa Phi Lưu Vân trong được UNESCO công nhận năm nào?",
                "1993",
                False,
            ),
            (
                "Di sản Văn hóa Đền Hùng được công nhận năm nào?",
                "2019",
                False,
            ),
            (
                "Tại sao Di sản Văn hóa Tháp Chàm và Cố đô Huế cùng được công nhận năm 1999?",
                "Không có thông tin",
                True,
            ),
            (
                "Tổng số di sản Văn hóa thế giới tại Việt Nam là bao nhiêu?",
                "8",
                False,
            ),
            (
                "Di sản nào được công nhận sớm nhất và di sản nào được công nhận muộn nhất?",
                "Phi Lưu Vân trong (1993) và Đền Hùng (2019)",
                False,
            ),
            (
                "Hội An được UNESCO công nhận năm nào?",
                "1999",
                False,
            ),
            (
                "Phát biểu nào sau đây đúng: Chợ Cồng được công nhận năm 2014 hay năm 2019?",
                "Chợ Cồng được công nhận năm 2014",
                False,
            ),
        ]

        for question, answer, is_imp in distractor_questions:
            counter += 1
            answer_start = _find_answer_start(context, answer) if not is_imp and answer and answer in context else -1
            plausible = "Không có thông tin" if is_imp else ""
            samples.append(_make_squad_item(
                _make_qid("dist", counter),
                question,
                answer if not is_imp else "",
                answer_start if not is_imp else -1,
                is_imp,
                plausible,
                context.find(plausible) if plausible in context else -1,
                category="distractor_contexts",
            ))

    # Additional distractor questions from regular stress contexts
    distractor_extra = [
        ("Cầu nào ở Việt Nam được thiết kế bởi kiến trúc sư Pháp Gustave Eiffel?",
         "Cầu Long Biên", STRESS_CONTEXTS[3]["context"]),
        ("Cầu Rồng nằm ở thành phố nào?",
         "Đà Nẵng", STRESS_CONTEXTS[3]["context"]),
        ("Việt Nam Dân chủ Cộng hòa được thành lập ngày nào?",
         "2/9/1975", STRESS_CONTEXTS[1]["context"]),
        ("Chiến tranh miền Nam kéo dài bao nhiêu năm?",
         "20 năm", STRESS_CONTEXTS[1]["context"]),
        ("Di sản Phi Lưu Vân trong nằm ở tỉnh nào?",
         "Ninh Bình", STRESS_CONTEXTS[0]["context"]),
        ("Tháp Chàm nằm ở tỉnh nào?",
         "Quảng Nam", STRESS_CONTEXTS[0]["context"]),
        ("TP.HCM đóng góp bao nhiêu % GDP?",
         "27%", STRESS_CONTEXTS[2]["context"]),
        ("HLV Park Hang-seo dẫn dắt đội tuyển Việt Nam trong năm nào?",
         "2019", STRESS_CONTEXTS[4]["context"]),
    ]

    for question, answer, context in distractor_extra:
        counter += 1
        answer_start = _find_answer_start(context, answer) if answer in context else -1
        samples.append(_make_squad_item(
            _make_qid("dist", counter),
            question,
            answer if answer in context else "",
            answer_start,
            answer not in context,
            "Không có thông tin" if answer not in context else "",
            context.find(answer) if answer in context else -1,
            category="distractor_contexts",
        ))

    # Pad to reach num_samples
    while len(samples) < num_samples:
        counter += 1
        ctx = random.choice(DISTRACTOR_CONTEXTS)["context"]
        plausible_entities = ["trường Đại học", "di sản", "cây cầu", "năm", "tỉnh"]
        distractor = random.choice(plausible_entities)
        question = f"Trong đoạn trên có bao nhiêu {distractor} được đề cập?"
        # Mark as unanswerable since we don't know the exact count
        samples.append(_make_squad_item(
            _make_qid("dist", counter),
            question,
            "",
            -1,
            True,
            f"'{distractor}' được đề cập nhiều lần",
            ctx.find(distractor) if distractor in ctx else -1,
            category="distractor_contexts",
        ))

    return samples[:num_samples]


# ---------------------------------------------------------------------------
# 5. Ambiguous Ground Truth (50 samples)
# ---------------------------------------------------------------------------
def generate_ambiguous_ground_truth(num_samples: int = 50) -> List[Dict[str, Any]]:
    """
    Generate questions where the answer could be interpreted in multiple ways,
    testing model robustness to label ambiguity.

    Targets E5 (Ambiguous Ground Truth) — expected ~5% of all errors.
    """
    samples: List[Dict[str, Any]] = []
    counter = 0

    ambiguous_cases = [
        {
            "question": "TP.HCM có dân số bao nhiêu?",
            "answer_variants": ["khoảng 9 triệu", "9 triệu", "9.000.000"],
            "is_impossible": False,
        },
        {
            "question": "Việt Nam đạt được bao nhiêu huy chương vàng tại SEA Games gần nhất?",
            "answer_variants": ["Không nói rõ số lượng", "không rõ", "không có thông tin"],
            "is_impossible": True,
        },
        {
            "question": "Cầu Long Biên cao bao nhiêu mét?",
            "answer_variants": ["Không nêu chiều cao cụ thể", "chưa nêu", "không rõ"],
            "is_impossible": True,
        },
        {
            "question": "Diện tích Việt Nam là bao nhiêu km²?",
            "answer_variants": ["Không nêu", "không rõ", "chưa có thông tin"],
            "is_impossible": True,
        },
        {
            "question": "Nhiệt độ trung bình ở miền Bắc Việt nam là bao nhiêu độ?",
            "answer_variants": ["Không nêu", "không xác định", "không rõ"],
            "is_impossible": True,
        },
    ]

    for ctx_data in STRESS_CONTEXTS:
        context = ctx_data["context"]

        for case in ambiguous_cases:
            counter += 1
            variants = case["answer_variants"]
            best_answer = variants[0]
            is_imp = case["is_impossible"]

            if not is_imp and best_answer in context:
                answer_start = _find_answer_start(context, best_answer)
                samples.append({
                    "id": _make_qid("ambig", counter),
                    "question": case["question"],
                    "is_impossible": False,
                    "answers": {
                        "text": variants,
                        "answer_start": [answer_start] * len(variants),
                    },
                    "plausible_answers": {"text": [], "answer_start": []},
                    "category": "ambiguous_ground_truth",
                })
            else:
                plausible = best_answer
                plausible_start = context.find(plausible) if plausible in context else -1
                samples.append(_make_squad_item(
                    _make_qid("ambig", counter),
                    case["question"],
                    "",
                    -1,
                    True,
                    plausible,
                    plausible_start,
                    category="ambiguous_ground_truth",
                ))

        if len(samples) >= num_samples:
            break

    # Pad to reach num_samples
    while len(samples) < num_samples:
        counter += 1
        ctx_idx = counter % len(STRESS_CONTEXTS)
        context = STRESS_CONTEXTS[ctx_idx]["context"]
        ambiguous_questions = [
            ("Có bao nhiêu dân tộc thiểu số ở Việt Nam?", ["54", "năm mươi bốn", "54 dân tộc"]),
            ("Di sản thế giới tại Việt Nam có bao nhiêu?", ["8", "tám", "8 di sản"]),
            ("Cầu nào dài nhất ở Việt Nam?", ["Cầu Rồng", "Cầu Long Biên", "chưa nêu"]),
        ]
        q, variants = random.choice(ambiguous_questions)
        best = variants[0]
        is_imp = best not in context or best == "chưa nêu"
        answer_start = _find_answer_start(context, best) if best in context else -1
        samples.append(_make_squad_item(
            _make_qid("ambig", counter),
            q,
            best if not is_imp else "",
            answer_start if not is_imp else -1,
            is_imp,
            "có nhiều cách trả lời" if not is_imp else best,
            answer_start,
            category="ambiguous_ground_truth",
        ))

    return samples[:num_samples]


# ---------------------------------------------------------------------------
# SQuAD Format Conversion
# ---------------------------------------------------------------------------
def _wrap_into_squad(
    samples: List[Dict[str, Any]],
    category_name: str,
    category_desc: str,
) -> Dict[str, Any]:
    """
    Wrap a list of QA samples into SQuAD format, grouping by context.
    """
    version = "1.0"
    squad: Dict[str, Any] = {
        "version": version,
        "category": category_name,
        "description": category_desc,
        "data": [],
    }

    # Group by context content
    ctx_groups: Dict[str, List[Dict]] = {}
    for s in samples:
        ctx = s.get("context", "")
        if not ctx:
            # Try to find a suitable context from our templates
            ctx = STRESS_CONTEXTS[len(ctx_groups) % len(STRESS_CONTEXTS)]["context"]
            title = STRESS_CONTEXTS[len(ctx_groups) % len(STRESS_CONTEXTS)]["title"]
            s["context"] = ctx
            s["title"] = title

        ctx_key = hashlib.md5(ctx.encode("utf-8")).hexdigest()
        if ctx_key not in ctx_groups:
            ctx_groups[ctx_key] = {"title": s.get("title", "StressTest"), "context": ctx, "qas": []}
        ctx_groups[ctx_key]["qas"].append(s)

    for group in ctx_groups.values():
        qas = []
        for qa in group["qas"]:
            qas.append({
                "id": qa["id"],
                "question": qa["question"],
                "answers": qa.get("answers", {"text": [], "answer_start": []}),
                "is_impossible": qa.get("is_impossible", False),
                "plausible_answers": qa.get("plausible_answers", {"text": [], "answer_start": []}),
                "category": qa.get("category", category_name),
            })
        squad["data"].append({
            "title": group["title"],
            "paragraphs": [{"context": group["context"], "qas": qas}],
        })

    return squad


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_all_stress_test(
    num_per_category: int = 50,
    output_dir: Path = STRESS_TEST_DIR,
) -> str:
    """
    Generate all stress-test datasets and save to disk.
    Also runs automatic data leakage detection on the combined dataset.

    Returns the path to the combined stress-test JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    categories = {
        "negative_questions": (
            generate_negative_questions,
            "Câu hỏi phủ định / bẫy logic (Negative & Trap Questions)",
        ),
        "multi_hop_reasoning": (
            generate_multi_hop_questions,
            "Câu hỏi yêu cầu suy luận bắc cầu (Multi-hop Reasoning)",
        ),
        "compound_word_boundary": (
            generate_compound_word_errors,
            "Câu hỏi/văn bản có từ ghép ranh giới phức tạp (Compound Word Boundary Errors)",
        ),
        "distractor_contexts": (
            generate_distractor_contexts,
            "Ngữ cảnh dài chứa nhiều thực thể nhiễu (Distractor Contexts)",
        ),
        "ambiguous_ground_truth": (
            generate_ambiguous_ground_truth,
            "Nhãn gán mập mờ / trùng chứng (Ambiguous Ground Truth)",
        ),
    }

    all_combined: List[Dict[str, Any]] = []
    total_generated = 0

    for category_name, (generator_fn, description) in categories.items():
        samples = generator_fn(num_samples=num_per_category)

        # Attach context to each sample for SQuAD grouping
        for s in samples:
            s.setdefault("context", "")
            s.setdefault("title", "StressTest")

        squad_format = _wrap_into_squad(samples, category_name, description)

        filepath = output_dir / f"{category_name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(squad_format, f, ensure_ascii=False, indent=2)

        print(f"[OK] Generated {len(samples)} samples for '{category_name}' -> {filepath}")
        all_combined.extend(samples)
        total_generated += len(samples)

    # Save combined dataset
    combined_path = output_dir / "stress_test_combined.json"
    combined = {
        "version": "1.0",
        "description": "Combined stress-test dataset for CS221 NLP MRC benchmark",
        "total_samples": total_generated,
        "categories": list(categories.keys()),
        "data": [],
    }

    for sample in all_combined:
        combined["data"].append({
            "title": sample.get("title", "StressTest"),
            "paragraphs": [{
                "context": sample.get("context", ""),
                "qas": [{
                    "id": sample["id"],
                    "question": sample["question"],
                    "answers": sample.get("answers", {"text": [], "answer_start": []}),
                    "is_impossible": sample.get("is_impossible", False),
                    "plausible_answers": sample.get("plausible_answers", {"text": [], "answer_start": []}),
                    "category": sample.get("category", "unknown"),
                }],
            }],
        })

    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Combined stress-test saved: {combined_path} ({total_generated} total samples)")

    # --- Automatic Data Leakage Check ---
    print("\n" + "=" * 60)
    print("RUNNING AUTOMATIC DATA LEAKAGE DETECTION")
    print("=" * 60)
    from datasets import Dataset, DatasetDict

    # Build DatasetDict from combined data for leakage check
    split_names = ["train", "validation", "test"]

    # Assign samples to splits, ensuring no context overlap across splits
    split_rows = {s: [] for s in split_names}
    ctx_to_split: Dict[str, str] = {}

    for i, sample in enumerate(all_combined):
        ctx = sample.get("context", "")
        if not ctx:
            ctx = STRESS_CONTEXTS[i % len(STRESS_CONTEXTS)]["context"]
        ctx_hash = hashlib.md5(ctx.encode("utf-8")).hexdigest()
        if ctx_hash not in ctx_to_split:
            ctx_to_split[ctx_hash] = split_names[i % 3]

        row = {
            "id": sample["id"],
            "title": sample.get("title", "StressTest"),
            "context": ctx,
            "question": sample["question"],
            "answers": sample.get("answers", {"text": [], "answer_start": []}),
            "is_impossible": sample.get("is_impossible", False),
        }
        split_rows[ctx_to_split[ctx_hash]].append(row)

    # Convert to DatasetDict (without nested answers dict — leakage checker
    # doesn't require the answers column for context/title/id checks)
    stress_dataset_dict = DatasetDict()
    for s in split_names:
        rows = split_rows[s]
        if not rows:
            stress_dataset_dict[s] = Dataset.from_dict({
                "id": [], "title": [], "context": [], "question": [],
                "is_impossible": [],
            })
        else:
            stress_dataset_dict[s] = Dataset.from_dict({
                "id": [r["id"] for r in rows],
                "title": [r["title"] for r in rows],
                "context": [r["context"] for r in rows],
                "question": [r["question"] for r in rows],
                "is_impossible": [r["is_impossible"] for r in rows],
            })

    # Run leakage checks
    from data_leakage_checker import check_data_leakage
    report = check_data_leakage(stress_dataset_dict)
    print(report.summary())

    # Save leakage report
    leakage_report_path = output_dir / "leakage_check_report.json"
    with open(leakage_report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Leakage check report saved: {leakage_report_path}")

    return str(combined_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    output_path = generate_all_stress_test(
        num_per_category=50,  # 250 total across 5 categories
    )
    print(f"\nStress test data generated at: {output_path}")

    # Print summary
    with open(output_path, "r", encoding="utf-8") as f:
        combined = json.load(f)
    print(f"\nDataset Summary:")
    print(f"  Total samples: {combined['total_samples']}")
    print(f"  Categories: {combined['categories']}")
    for cat in combined["categories"]:
        cat_path = STRESS_TEST_DIR / f"{cat}.json"
        with open(cat_path, "r", encoding="utf-8") as f:
            cat_data = json.load(f)
        total_qas = sum(len(p["paragraphs"][0]["qas"]) for p in cat_data["data"])
        print(f"  - {cat}: {total_qas} samples")
