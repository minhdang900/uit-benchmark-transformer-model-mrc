# A Diagnostic Stress-Test for Vietnamese Extractive MRC

Bộ dữ liệu chẩn đoán và quy trình phân tích lỗi cho bài toán **đọc hiểu & trả lời câu hỏi tiếng Việt** (Vietnamese Extractive Machine Reading Comprehension).

**Môn học:** CS221 — Xử lý Ngôn ngữ Tự nhiên · Trường ĐH Công nghệ Thông tin, ĐHQG-HCM
**GVHD:** ThS. Đặng Văn Thìn

---

## Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Bộ dữ liệu stress-test (5 nhóm lỗi) | ✅ **Hoàn thành** — 250 cặp Q-A |
| Kiểm tra rò rỉ dữ liệu (leakage check) | ✅ Hoàn thành |
| Pipeline huấn luyện & đánh giá | ✅ Đã cài đặt · ⏳ **chưa chạy** |
| Kết quả thực nghiệm | ⏳ **Chưa đo** — xem `results/README.md` |

> **Nguyên tắc của repo này:** không con số nào xuất hiện trong README, báo cáo hay slide nếu không truy vết được tới một file trong `results/`.

---

## Vấn đề nghiên cứu

Các bộ benchmark MRC tiếng Việt hiện tại báo cáo **một điểm tổng** (EM / F1) trên tập test chuẩn. Điểm tổng cho biết mô hình nào cao hơn, nhưng **không cho biết mô hình sai ở đâu và vì sao** — nên không giúp gì cho việc cải tiến.

Với tiếng Việt, câu hỏi "sai ở đâu" đặc biệt quan trọng vì **~85% từ vựng là từ ghép đa âm tiết**. Ranh giới từ là một nguồn lỗi có thật, nhưng bị điểm tổng che khuất hoàn toàn.

## Đóng góp

1. **Bộ stress-test chẩn đoán** — 250 cặp Q-A chia theo **5 nhóm lỗi ngôn ngữ học**, mỗi nhóm cô lập một cơ chế thất bại
2. **Quy trình kiểm tra rò rỉ dữ liệu** giữa stress-test và tập huấn luyện
3. **Khung phân tích lỗi** áp dụng lên tương phản *monolingual vs multilingual* — hai họ mô hình khác nhau **đúng ở khâu tokenization**, thứ mà nhóm lỗi E1 được thiết kế để dò

Trọng tâm là **công cụ chẩn đoán**, không phải bảng xếp hạng. Mô hình ở đây đóng vai trò *đối tượng thử nghiệm* để chứng minh công cụ hoạt động.

---

## Bộ dữ liệu stress-test

`data/stress_test/` — 5 nhóm × 50 câu = **250 cặp Q-A**

| Mã | Nhóm | Cơ chế thất bại được dò | Answerable | Unanswerable |
|---|---|---|---|---|
| **E1** | `compound_word_boundary` | Ranh giới từ ghép tiếng Việt | 41 | 9 |
| **E2** | `distractor_contexts` | Ngữ cảnh dài, nhiều thực thể nhiễu | 35 | 15 |
| **E3** | `negative_questions` | Câu hỏi phủ định / bẫy logic | **0** ⚠️ | 50 |
| **E4** | `multi_hop_reasoning` | Suy luận bắc cầu qua nhiều sự kiện | 37 | 13 |
| **E5** | `ambiguous_ground_truth` | Nhãn nhập nhằng | **7** ⚠️ | 43 |

### ⚠️ Hai vấn đề đã biết của bộ dữ liệu

**1. E3 và E5 mất cân bằng answerable/unanswerable.**
E3 có **100%** câu unanswerable, E5 có **86%**. Với extractive MRC, một mô hình **luôn trả lời "không có đáp án"** sẽ đạt điểm tuyệt đối ở E3 mà không chứng minh được năng lực nào. Hai nhóm này **cần bổ sung câu answerable làm đối chứng** thì mới có giá trị chẩn đoán.

**2. `stress_test_combined.json` chứa 370 bài viết nhưng chỉ 250 cặp Q-A.**
Tổng 5 nhóm rời là 145 bài viết / 250 Q-A. Cần đối chiếu lại xem 225 bài viết dư là ngữ cảnh nhiễu cố ý hay lỗi gộp file.

Cả hai điểm này phải được nêu trong phần **Limitations** của báo cáo.

---

## Cấu trúc

```
├── src/
│   ├── stress_test_generator.py   Sinh bộ dữ liệu chẩn đoán
│   ├── data_leakage_checker.py    Kiểm tra rò rỉ với tập train
│   ├── dataset_loader.py          Nạp UIT-ViQuAD 2.0 + stress-test
│   ├── baseline_bm25.py           Baseline không-neural
│   ├── train_mrc_transformer.py   Fine-tune & đánh giá
│   ├── eval_metrics.py            Exact Match / F1
│   └── error_analysis.py          Phân tích lỗi theo nhóm
├── data/stress_test/              Bộ dữ liệu (đã hoàn thành)
├── results/                       Kết quả đo (chưa có)
├── reports/                       Báo cáo dữ liệu
└── docs/                          Đề cương & slide
```

## Chạy

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Baseline không-neural (CPU)
python src/baseline_bm25.py

# 2. Fine-tune một mô hình
python src/train_mrc_transformer.py --model_name vinai/phobert-base-v2 --epochs 3 --seed 42

# 3. Đánh giá trên stress-test theo từng nhóm lỗi
python src/error_analysis.py
```

Sau lần chạy thành công đầu tiên: `pip freeze > requirements.lock.txt`

---

## Nhóm thực hiện

| Họ tên | MSSV | Email |
|---|---|---|
| Nguyễn Quang Lâm | 25210289 | 25210289@ms.uit.edu.vn |
| Trần Trọng Tấn | 25210334 | 25210334@ms.uit.edu.vn |
| Lê Quang Thi | 25210337 | 25210337@ms.uit.edu.vn |
| Vỏ Cẩm Thu | 25210342 | 25210342@ms.uit.edu.vn |
