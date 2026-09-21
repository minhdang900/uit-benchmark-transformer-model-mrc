# Đề cương nghiên cứu / Problem Statement

**Đề tài:** A Diagnostic Stress-Test for Vietnamese Extractive Machine Reading Comprehension
**Môn:** CS221 — Xử lý Ngôn ngữ Tự nhiên · GVHD: ThS. Đặng Văn Thìn

---

## 0. Vì sao phải đổi khung tiếp cận

Khung ban đầu của nhóm là **"khảo sát & đánh giá hiệu năng các mô hình Transformer"** — tức một **benchmark**. Khung này có ba điểm yếu:

| Điểm yếu | Hệ quả |
|---|---|
| Benchmark không phải "phát triển phương pháp" | Thầy nêu rõ ở Buổi 1: *"mình muốn là mọi người **phát triển phương pháp** hơn là cái dữ liệu"*. Chạy 5 mô hình pre-trained với config mặc định không tạo ra phương pháp mới |
| Đóng góp = một cái bảng | Hội đồng hỏi *"em xây dựng cái gì?"* thì khó trả lời |
| Phụ thuộc hoàn toàn vào GPU | Thiếu một mô hình là thủng một ô trong bảng → nghiên cứu trông dở dang |

**Khung mới** giữ nguyên toàn bộ code và dữ liệu đã làm, chỉ đổi **điều mình tuyên bố chứng minh**:

> Thay vì *"chúng tôi so sánh 5 mô hình"*
> → *"chúng tôi xây một **công cụ chẩn đoán** phơi bày chỗ mô hình MRC tiếng Việt thất bại, và minh chứng nó trên tương phản monolingual vs multilingual."*

| | Khung benchmark | Khung chẩn đoán |
|---|---|---|
| Đóng góp | Bảng so sánh | **Bộ dữ liệu + phương pháp chẩn đoán** |
| Có phải "phương pháp" không? | Không | **Có** |
| Số mô hình cần chạy | 5 | **2 là đủ** |
| Nếu chỉ chạy được 2 | Dở dang | **Vẫn là nghiên cứu hoàn chỉnh** |
| Gắn với nội dung môn học | Yếu (MRC không được dạy) | **Mạnh** — E1 chính là bài toán tách từ, chủ đề xuyên suốt môn |

---

## 1. Vấn đề (What)

Cho một câu hỏi *Q* và một đoạn văn *C* bằng tiếng Việt, hệ thống phải **trích xuất đúng ranh giới đoạn văn chứa câu trả lời** *A ⊆ C*, hoặc xác định *C* không chứa câu trả lời.

Các benchmark MRC tiếng Việt hiện tại đánh giá bằng **một điểm tổng** EM / F1 trên tập test chuẩn.

## 2. Khoảng trống (Gap)

Điểm tổng trả lời được *"mô hình nào cao hơn"* nhưng **không trả lời được** *"mô hình sai ở đâu, và tại sao"*.

Ba hệ quả cụ thể:

1. **Không định vị được lỗi.** Hai mô hình cùng đạt 70% EM có thể sai ở hai nhóm hiện tượng hoàn toàn khác nhau.
2. **Che khuất lỗi đặc thù tiếng Việt.** ~85% từ vựng tiếng Việt là từ ghép đa âm tiết. Mô hình cắt đúng nội dung nhưng lệch ranh giới từ ghép sẽ mất điểm EM mà không ai biết nguyên nhân.
3. **Không hướng dẫn được cải tiến.** Biết mô hình đạt 70% không cho biết nên sửa gì tiếp theo.

## 3. Tại sao khoảng trống này đáng giải quyết (Why)

- **Về mặt khoa học:** tiếng Việt là **ngôn ngữ ít tài nguyên**; kết luận từ tiếng Anh không chuyển giao trực tiếp. Đặc điểm ngôn ngữ (từ ghép, không biến hình) tạo ra các cơ chế lỗi mà benchmark tiếng Anh không có.
- **Về mặt thực tiễn:** người triển khai cần biết mô hình hỏng ở loại đầu vào nào để quyết định khi nào cần fallback sang người.
- **Về mặt phương pháp:** đây là bước **error analysis** mà giảng viên nhấn mạnh xuyên suốt Buổi 3, 5, 6, 7 như phần quan trọng nhất của một nghiên cứu NLP.

## 4. Câu hỏi nghiên cứu

Đặt dạng **Yes/No** để dễ kiểm chứng, theo đúng hướng dẫn Buổi 3.

| # | Câu hỏi | Nhóm lỗi liên quan |
|---|---|---|
| **RQ1** | Mô hình **word-level** (PhoBERT) có nhạy cảm với lỗi ranh giới từ hơn mô hình **subword** (XLM-R) không? | E1 |
| **RQ2** | Khi ngữ cảnh dài ra và nhiều thực thể nhiễu, hiệu năng có suy giảm khác nhau giữa hai họ mô hình không? | E2 |
| **RQ3** | Mô hình có thực sự "hiểu" phủ định, hay chỉ học thiên lệch về việc từ chối trả lời? | E3 |
| **RQ4** | Transformer có vượt baseline không-neural ở **mọi** nhóm lỗi, hay chỉ ở một số nhóm? | Tất cả |

**Giả thuyết (RQ1):** PhoBERT yếu hơn ở E1 vì tokenizer word-level buộc phải tách từ trước, nên **lỗi tách từ lan truyền** vào mô hình; XLM-R dùng SentencePiece nên không phụ thuộc bước này.

Đây là **tương phản được thiết kế**, không phải so sánh ngẫu nhiên: hai mô hình được chọn vì chúng khác nhau **đúng ở khâu tokenization** — thứ mà E1 được xây để dò.

## 5. Phương pháp giải quyết (How)

### 5.1 Công cụ chẩn đoán
Bộ stress-test 250 cặp Q-A, 5 nhóm × 50 câu, mỗi nhóm **cô lập một cơ chế thất bại** (E1–E5). Kèm quy trình kiểm tra rò rỉ dữ liệu với tập huấn luyện.

> **Cập nhật 2026-09-21:** bộ 250 câu này đã được kiểm định, thấy không dùng được (chỉ 15/120 câu có đáp án chứa đáp án trong ngữ cảnh) và đã gỡ khỏi kho mã. Công cụ chẩn đoán hiện tại là **bộ stress-test v2** (1.139 mục, 572 ngữ cảnh, dựng từ validation, kiểm định tự động 0 vi phạm) — xem `data/stress_test_v2/README.md`.

### 5.2 Đối tượng thử nghiệm

| Ưu tiên | Mô hình | Vai trò |
|---|---|---|
| 1 | **BM25 / TF-IDF** | Baseline không-neural, chạy CPU — định lượng phần transformer thực sự đóng góp |
| 2 | **PhoBERT-base-v2** | Nhánh monolingual, tokenizer **word-level** |
| 3 | **XLM-R-base** | Nhánh multilingual, tokenizer **subword** |
| — | mBERT, ViDeBERTa | **Tùy chọn** — chỉ chạy nếu còn thời gian |

### 5.3 Quy trình đánh giá
```
Train trên UIT-ViQuAD 2.0 (seed cố định = 42)
  → Tune hyperparameter trên tập DEV
    → Đánh giá 1 lần trên tập TEST
      → Đánh giá riêng từng nhóm E1–E5 trên stress-test
        → Phân tích lỗi thủ công 100 mẫu, phân nhóm nguyên nhân
```

### 5.4 Bảng kết quả chính

Không phải bảng xếp hạng, mà là **ma trận chẩn đoán**:

| Mô hình | ViQuAD test | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|---|
| BM25 | | | | | | |
| PhoBERT | | | | | | |
| XLM-R | | | | | | |

Giá trị nằm ở **hình dạng của các chênh lệch giữa các cột**, không phải ở con số cao nhất.

---

## 6. Hạn chế đã biết — phải nêu trong báo cáo

> **Cập nhật 2026-09-21:** mục 1–3 là hạn chế của bộ v1 và đã được bộ v2 giải quyết — E3 đọc theo cặp với câu gốc đi kèm nên "luôn từ chối" không thắng được; tệp gộp 370/250 không còn (v1 đã gỡ); mỗi nhóm v2 có 120–223 mục thay vì 50. Mục 4 vẫn đúng: báo cáo dùng hai seed và nêu rõ điều đó.

1. **E3 có 100% câu unanswerable, E5 có 86%.** Mô hình luôn từ chối trả lời sẽ đạt điểm tuyệt đối ở E3 mà không chứng minh năng lực gì. Hai nhóm này cần bổ sung câu answerable làm đối chứng; hiện chưa có.
2. **`stress_test_combined.json` có 370 bài viết nhưng chỉ 250 cặp Q-A** — cần đối chiếu lại việc gộp file.
3. **Quy mô 50 mẫu/nhóm** đủ để quan sát xu hướng, chưa đủ cho kết luận thống kê mạnh.
4. Nếu chỉ chạy được **một seed**, phải nêu rõ và không tuyên bố chênh lệch nhỏ là có ý nghĩa.
5. Dữ liệu gốc từ Wikipedia — **chưa phủ** miền mạng xã hội, nơi lỗi tách từ còn nặng hơn.

---

## 7. Ánh xạ sang nội dung môn học

| Nội dung được dạy | Dùng ở đâu trong đề tài |
|---|---|
| Buổi 3 — Tiền xử lý, tách từ | Nhóm lỗi **E1**; pipeline cho PhoBERT |
| Buổi 3 — Quy trình nghiên cứu 6 bước | Cấu trúc toàn bộ đề cương này |
| Buổi 4 — TF-IDF, biểu diễn văn bản | **Baseline BM25** |
| Buổi 5 — Phân tích lỗi, confusion analysis | Phần phân tích lỗi thủ công |
| Buổi 6 — Ít dữ liệu thì ML truyền thống thắng | Lý do giữ baseline không-neural |
| Buổi 7 — Giới hạn max length; word vs syllable | Chọn mô hình; phân tích **E2** |
| Buổi 7 — Quy trình gán nhãn & độ đồng thuận | Quy trình xây stress-test |

> Luận điểm xuyên suốt của môn học — *"phải dựa trên **đặc điểm dữ liệu** để chọn phương pháp"* — chính là kết luận mà bộ công cụ chẩn đoán này được thiết kế để chứng minh bằng số liệu.
