# PhoBERT vs XLM-R cho đọc hiểu trích xuất tiếng Việt — một đánh giá chẩn đoán

**Đề tài:** Khảo sát và đánh giá hiệu năng các mô hình Transformer tiền huấn luyện trong bài toán
đọc hiểu văn bản và trả lời câu hỏi tiếng Việt
**Môn:** CS221 — Xử lý ngôn ngữ tự nhiên · UIT, ĐHQG-HCM · **GVHD:** TS. Đặng Văn Thìn

> **Nguyên tắc:** không con số nào trong README, báo cáo hay slide mà không truy vết được tới một tệp
> trong `results/`. Báo cáo và slide đọc số từ `report/generated/`, do `scripts/make_report_numbers.py`
> sinh ra từ `results/` — không có số gõ tay.

## Sản phẩm

| | Tệp |
|---|---|
| Báo cáo (PDF) | `report/main.pdf` — dựng: `cd report && latexmk -xelatex main.tex` |
| Slide (PDF) | `slides/main.pdf` — dựng: `cd slides && latexmk -xelatex main.tex` |
| Kết quả, dự đoán từng câu, nhật ký | `results/` |

## Kết quả chính

UIT-ViQuAD 2.0, **toàn bộ validation (3.814 câu)**, chấm một lần. Test chính thức ẩn nhãn nên không
chấm được; epoch được chọn trên tập dev tách từ train. Nguồn: `results/eval_*_validation.json`,
`results/diagnosis_validation.json`.

| Hệ thống | EM | F1 | EM câu có đáp án | EM câu không có đáp án | Tỉ lệ từ chối |
|---|---:|---:|---:|---:|---:|
| Luôn từ chối | 30,44 | 30,44 | 0,00 | 100,00 | 100,00 |
| TF-IDF (truy hồi câu) | 1,05 | 22,11 | 1,51 | 0,00 | 0,00 |
| XLM-R-base (fine-tune) | 52,49 | 61,32 | 52,02 | 53,57 | 32,96 |
| PhoBERT-base-v2 (fine-tune) | 51,36 | 63,10 | **66,91** | 15,85 | 5,66 |

**Điều bảng này nói ra mà điểm tổng thì không:**

1. **Điểm tổng như nhau** — chênh lệch EM cặp −1,13, CI 95% [−2,88; 0,60], McNemar p = 0,223.
2. **Hai hồ sơ kỹ năng ngược nhau** — PhoBERT hơn **+14,89** EM trên câu có đáp án; XLM-R hơn
   **37,73** EM trên câu không có đáp án.
3. **Tách từ không phải yếu tố quyết định** — chỉ 1,24% đáp án validation cắt ngang một từ của
   `pyvi` (trần EM của mô hình word-level hoàn hảo: 99,25%), và phần lớn các trường hợp đó là lỗi
   của bộ tách từ. Khi đã trả lời, tỉ lệ lỗi biên của hai mô hình như nhau (27,62% và 26,86%).
4. **PhoBERT bỏ lớp "không có đáp án"** — loss trên nhãn "không có đáp án" tăng ~7 lần sau epoch 1
   trong khi loss trên vị trí đáp án vẫn giảm; lặp lại ở lr 3e-5 và 2e-5 (`results/loss_probe.json`).
5. **Bộ stress-test 250 câu của nhóm không dùng được làm thước đo** — chỉ 15/120 câu có đáp án chứa
   đáp án trong ngữ cảnh; "luôn từ chối" đứng đầu bảng điểm trên bộ này
   (`results/stress_test_audit.json`).

## Tái lập

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
python scripts/fetch_data.py                      # UIT-ViQuAD 2.0 -> data/raw/
python -m pytest -m "not slow"                    # 218 kiểm thử, không cần GPU

python scripts/analyze_segmentation.py            # biên từ + trần EM (không cần mô hình)
python scripts/audit_stress_test.py               # kiểm định stress-test

python scripts/finetune.py --model FacebookAI/xlm-roberta-base --out models/xlmr \
    --epochs 3 --max-length 384 --doc-stride 128 --max-answer-len 64
python scripts/finetune.py --model vinai/phobert-base-v2 --out models/phobert --word-segmented \
    --epochs 3 --max-length 256 --doc-stride 96 --max-answer-len 64

python scripts/run_eval.py --models abstain baseline xlmr phobert
python scripts/run_eval.py --models abstain baseline xlmr phobert --dataset stress
python scripts/diagnose.py && python scripts/probe_loss.py
python scripts/make_report_numbers.py && python scripts/make_figures.py
```

Thiết bị đã dùng: Apple M5 Pro (MPS). Một epoch XLM-R ≈ 35 phút, PhoBERT ≈ 22 phút.

## Cấu trúc

```
src/mrc/        pipeline có kiểm thử (kế thừa đồ án CS116 của cùng nhóm) + phần CS221:
                segmented_tokenizer.py  PhoBERT: tách từ pyvi + BPE + offset về văn bản gốc
                segmentation.py         tương thích biên từ, trần EM
                stress_test.py          nạp + kiểm định stress-test
                diagnosis.py            phân loại lỗi, từ chối, McNemar, bootstrap
scripts/        fine-tune, chấm, chẩn đoán, sinh số liệu và hình cho báo cáo
results/        mọi con số: eval_*, predictions_*, training_curve_*, diagnosis_*, logs/
report/         mã LaTeX báo cáo (generated/ do script sinh)
slides/         mã Beamer, dùng chung số liệu với báo cáo
legacy/         khung mã ban đầu — chưa từng chạy, không dùng (xem legacy/README.md)
```

## Nhóm thực hiện

| Họ tên | MSSV | Email |
|---|---|---|
| Nguyễn Quang Lâm | 25210289 | 25210289@ms.uit.edu.vn |
| Trần Trọng Tấn | 25210334 | 25210334@ms.uit.edu.vn |
| Lê Quang Thi | 25210337 | 25210337@ms.uit.edu.vn |
| Vỏ Cẩm Thu | 25210342 | 25210342@ms.uit.edu.vn |
