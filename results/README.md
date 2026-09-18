# results/

Mọi con số trong README, báo cáo (`report/`) và slide (`slides/`) đến từ các tệp ở đây.

| Tệp | Sinh bởi | Nội dung |
|---|---|---|
| `eval_<hệ thống>_<validation\|stress>.json` | `scripts/run_eval.py` | EM/F1 tổng, tách có/không đáp án, theo độ dài, provenance (commit, thiết bị, thời điểm) |
| `predictions_<hệ thống>_<tập>.json` | `scripts/run_eval.py` | dự đoán từng câu `{qid: đáp án}` |
| `training_curve_<run>.json` | `scripts/finetune.py` | cấu hình, loss và EM/F1 dev theo epoch, epoch được chọn |
| `segmentation_validation.json` | `scripts/analyze_segmentation.py` | tương thích biên từ và trần EM từng câu |
| `misaligned_labels.json` | phân loại thủ công | nguyên nhân của 33 câu lệch biên (cần nhóm rà lại) |
| `stress_test_audit.json` | `scripts/audit_stress_test.py` | kiểm định bộ stress-test |
| `diagnosis_validation.json` | `scripts/diagnose.py` | phân loại lỗi, từ chối, so sánh cặp, bẫy plausible |
| `disagreements_phobert_xlmr.json`, `error_sample_*.json` | `scripts/diagnose.py` | nguyên liệu phân tích định tính |
| `loss_probe.json` | `scripts/probe_loss.py` | loss theo loại nhãn cho từng checkpoint |
| `logs/` | nhật ký huấn luyện đã lọc | tiến trình loss theo bước |

Hệ thống: `abstain` (luôn từ chối), `baseline` (TF-IDF), `xlmr`, `phobert` (lr 3e-5, lần chạy chính),
`phobert_lr2e5` (lần chạy lại). Lần chạy PhoBERT chính được chọn theo F1 dev (`scripts/_runs.py`).
