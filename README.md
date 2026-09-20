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
Epoch, lần chạy và ngưỡng từ chối τ đều chọn trên dev tách từ train. Mỗi mô hình **hai seed, báo cáo ngang
nhau**. Nguồn: `results/eval_*_validation.json`, `results/thresholds.json`, `results/diagnosis_validation.json`.

| Hệ thống | Seed | τ (dev) | EM / F1 ở τ = 0 | EM / F1 ở τ dev | Từ chối ở τ dev |
|---|---:|---:|---:|---:|---:|
| Luôn từ chối | — | — | 30,44 / 30,44 | — | 100% |
| TF-IDF (truy hồi câu) | — | — | 1,05 / 22,11 | — | 0% |
| XLM-R-base | 42 | −0,47 | 52,49 / 61,32 | 52,49 / 61,99 | 27,8% |
| XLM-R-base | 13 | −0,62 | 53,64 / 62,18 | 53,25 / 62,68 | 24,3% |
| PhoBERT-base-v2 | 42 | +7,86 | 51,36 / 63,10 | **58,31 / 68,54** | 25,0% |
| PhoBERT-base-v2 | 13 | +1,03 | 58,94 / 68,91 | **59,26 / 68,49** | 32,1% |

1. **Ở τ chọn trên dev, PhoBERT tốt hơn XLM-R ở cả hai seed**: +5,82 EM (seed 42, CI theo bài viết
   [3,60; 7,72]) và +6,00 EM (seed 13, [3,30; 8,29]). Sau hiệu chỉnh, dao động theo seed < 1 điểm F1.
2. **Phần hơn đó là hiệu chỉnh, không phải khả năng đọc.** Khớp cả cấu hình cửa sổ lẫn tính ổn định của
   lần chạy (PhoBERT s13 vs XLM-R ở 256/96): chênh lệch trên câu **có đáp án** là **−0,08 EM**
   (CI theo bài viết [−2,24; 2,05], p = 0,97) — một kết quả "không khác biệt" đo chắc chắn, CI chỉ rộng
   ±2 điểm. Toàn bộ +5,85 EM nằm ở câu **không có đáp án** (+19,38). Lát cắt "cả hai cùng trả lời" đổi
   dấu theo lần chạy (+2,60 với seed 13; −0,84 với lr 1e-5), nên không dùng để kết luận.
3. **Seed 42 ở τ = 0 cho một kết luận sai**: bằng nhau về tổng (−1,13, p = 0,22) với "hai hồ sơ ngược nhau"
   (PhoBERT +14,89 EM có đáp án, XLM-R +37,73 không đáp án) — vì PhoBERT seed 42 chỉ từ chối 5,7%.
   Seed 13 không có hiện tượng này (+5,30 EM cho PhoBERT ngay ở τ = 0).
4. **PhoBERT seed 42 "không từ chối" là một sự kiện khi huấn luyện**, tái hiện được: huấn luyện tiếp từ epoch 1
   ở lr 2,2e-5 → chuẩn gradient vọt 12.880, loss "không có đáp án" 0,71 → 4,35; cùng batch ở lr 1e-5 thì không
   (`results/probe_resume.json`). Nhưng seed 13 ở cùng lr 3e-5 không sụp đổ → phụ thuộc lần chạy.
   Dữ liệu sạch: 0 vi phạm trên 31.039 feature.
5. **Cửa sổ không phải lợi thế của XLM-R**: ở 256/96 như PhoBERT, XLM-R đạt F1 63,03 ở τ = 0 — ngang hoặc
   nhỉnh hơn hai lần chạy 384/128.
6. **Tách từ không phải yếu tố quyết định** — chỉ 1,24% đáp án cắt ngang một từ của `pyvi` (trần 0,75 EM).
7. **Bộ stress-test 250 câu của nhóm không dùng được làm thước đo** — chỉ 15/120 câu có đáp án chứa đáp án.

Mốc tham chiếu (VLSP 2021 public test, Nguyen et al. 2022): baseline mBERT F1 63,03; đội cao nhất 84,24;
người 87,34.

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
demo/           ViMRC Console (Streamlit): chạy mô hình thật + đọc lại results/
                — phần thêm để trình bày, không thuộc sản phẩm nộp (demo/README.md)
```

## Nhóm thực hiện

| Họ tên | MSSV | Email |
|---|---|---|
| Nguyễn Quang Lâm | 25210289 | 25210289@ms.uit.edu.vn |
| Trần Trọng Tấn | 25210334 | 25210334@ms.uit.edu.vn |
| Lê Quang Thi | 25210337 | 25210337@ms.uit.edu.vn |
| Vỏ Cẩm Thu | 25210342 | 25210342@ms.uit.edu.vn |
