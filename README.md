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

UIT-ViQuAD 2.0, **toàn bộ validation (3.814 câu)** — chính là public test của VLSP 2021 — chấm một lần.
Epoch, lần chạy và ngưỡng từ chối τ đều được chọn trên dev tách từ train. Nguồn:
`results/eval_*_validation.json`, `results/thresholds.json`, `results/diagnosis_validation.json`.

| Hệ thống | τ | EM / F1 ở τ = 0 | EM / F1 ở τ chọn trên dev | EM có đáp án | EM không đáp án | Từ chối |
|---|---:|---:|---:|---:|---:|---:|
| Luôn từ chối | — | 30,44 / 30,44 | — | 0,00 | 100,00 | 100% |
| TF-IDF (truy hồi câu) | — | 1,05 / 22,11 | — | 1,51 | 0,00 | 0% |
| XLM-R-base | −0,47 | 52,49 / 61,32 | 52,49 / 61,99 | 54,62 | 47,63 | 27,8% |
| PhoBERT-base-v2 | +7,86 | 51,36 / 63,10 | **58,31 / 68,54** | 60,57 | 53,14 | 25,0% |

1. **Ở τ = 0, hai mô hình bằng nhau về tổng** (chênh EM −1,13, CI theo bài viết [−3,53; 0,91],
   p = 0,223) và trông như hai hồ sơ ngược nhau (PhoBERT +14,89 EM có đáp án, XLM-R +37,73 không đáp án).
   Nhưng PhoBERT chỉ từ chối 5,7%, XLM-R 33%: trên 2.011 câu cả hai cùng trả lời, khoảng cách "đọc" chỉ +4,1.
2. **Ở τ chọn trên dev, PhoBERT tốt hơn trên cả hai loại câu**: +5,82 EM tổng (CI theo bài viết
   [3,60; 7,72], p < 0,001), +5,96 trên câu có đáp án, +5,51 trên câu không có đáp án.
3. **PhoBERT "không từ chối" là một sự kiện khi huấn luyện**, tái hiện được: huấn luyện tiếp từ epoch 1
   ở lr 2,2e-5 → chuẩn gradient vọt tới 12.880, loss "không có đáp án" 0,71 → 4,35; cùng batch ở lr 1e-5
   thì không (`results/probe_resume.json`). Dữ liệu sạch: 0 vi phạm trên 31.039 feature.
   Khả năng phân biệt còn nguyên — chỉ điểm [CLS] bị dịch ~8 logit.
4. **Tách từ không phải yếu tố quyết định** — chỉ 1,24% đáp án cắt ngang một từ của `pyvi` (trần thiệt hại
   0,75 EM), phần lớn do lỗi của bộ tách từ.
5. **Bộ stress-test 250 câu của nhóm không dùng được làm thước đo** — chỉ 15/120 câu có đáp án chứa đáp án
   trong ngữ cảnh (`results/stress_test_audit.json`).

Mốc tham chiếu (VLSP 2021 public test, Nguyen et al. 2022): baseline mBERT F1 63,03; đội cao nhất 84,24;
người 87,34. Độ dao động theo seed và XLM-R ở 256/96: xem mục "seed" trong báo cáo.

## Tái lập

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
python scripts/fetch_data.py                      # UIT-ViQuAD 2.0 -> data/raw/
python -m pytest -m "not slow"                    # 244 kiểm thử, không cần GPU

python scripts/analyze_segmentation.py            # biên từ + trần EM (không cần mô hình)
python scripts/audit_stress_test.py               # kiểm định stress-test

python scripts/finetune.py --model FacebookAI/xlm-roberta-base --out models/xlmr \
    --epochs 3 --max-length 384 --doc-stride 128 --max-answer-len 64
python scripts/finetune.py --model vinai/phobert-base-v2 --out models/phobert --word-segmented \
    --epochs 3 --max-length 256 --doc-stride 96 --max-answer-len 64

python scripts/run_eval.py --models abstain baseline xlmr phobert
python scripts/run_eval.py --models abstain baseline xlmr phobert --dataset stress
python scripts/scan_features.py --model vinai/phobert-base-v2 --word-segmented \
    --max-length 256 --doc-stride 96 --name phobert          # quét feature
bash scripts/queue_score_windows.sh                         # điểm từng cửa sổ (dev + validation)
python scripts/calibrate_thresholds.py --runs xlmr phobert  # τ chọn trên dev, chấm validation 1 lần
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
