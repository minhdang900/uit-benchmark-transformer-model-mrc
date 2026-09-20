# results/

Mọi con số trong README, báo cáo (`report/`) và slide (`slides/`) đến từ các tệp ở đây.

| Tệp | Sinh bởi | Nội dung |
|---|---|---|
| `eval_<hệ thống>_<validation\|stress>.json` | `scripts/run_eval.py` | EM/F1 tổng, tách có/không đáp án, theo độ dài, provenance (commit, thiết bị, thời điểm) |
| `predictions_<hệ thống>_<tập>.json` | `scripts/run_eval.py` | dự đoán từng câu `{qid: đáp án}` |
| `training_curve_<run>.json` | `scripts/finetune.py` | cấu hình, loss và EM/F1 dev theo epoch, epoch được chọn |
| `segmentation_validation.json` | `scripts/analyze_segmentation.py` | tương thích biên từ và trần EM từng câu |
| `misaligned_labels.json` | phân loại thủ công | nguyên nhân của 33 câu lệch biên (cần nhóm rà lại) |
| `stress_v2_audit.json` | `scripts/build_stress_v2.py` | thiết kế + kiểm định bộ stress-test v2 (0 vi phạm) |
| `stress_v2_slice_scores.json` | `scripts/stress_v2_slice_scores.py` | điểm của các *slice* v2 chiếu lên dự đoán validation (không gồm perturbation) |
| `dataset_analysis.json` | `scripts/analyze_dataset.py` | thống kê mô tả UIT-ViQuAD 2.0 (độ dài, loại câu hỏi, phân bố nhãn) |
| `diagnosis_validation.json` | `scripts/diagnose.py` | phân loại lỗi, từ chối, so sánh cặp, bẫy plausible |
| `disagreements_phobert_xlmr.json`, `error_sample_*.json` | `scripts/diagnose.py` | nguyên liệu phân tích định tính |
| `loss_probe.json` | `scripts/probe_loss.py` | loss theo loại nhãn cho từng checkpoint |
| `logs/` | nhật ký huấn luyện đã lọc | tiến trình loss theo bước |
| `windows_<run>_<dev\|validation>.json` | `scripts/score_windows.py` | điểm từng cửa sổ (null − best) để dựng lại dự đoán ở mọi τ |
| `thresholds.json` | `scripts/calibrate_thresholds.py` | τ chọn trên dev, validation ở τ = 0 và τ đã chọn, câu cả hai cùng trả lời |
| `predictions_<run>_tuned_validation.json` | `scripts/calibrate_thresholds.py` | dự đoán ở τ chọn trên dev |
| `feature_scan_<model>.json` | `scripts/scan_features.py` | kiểm tra toàn bộ feature huấn luyện |
| `probe_resume.json`, `probe_probe_*.json` | `scripts/finetune.py --resume-from` | huấn luyện tiếp PhoBERT từ epoch 1 ở hai tốc độ học |
| `published_viquad2.json` | trích từ Nguyen et al. (2022) | kết quả đã công bố trên VLSP 2021 |
| `logs/steps_<run>.jsonl` | `scripts/finetune.py` | loss, loss theo nhãn, chuẩn gradient trước khi cắt, lr — từng bước |

Hệ thống: `abstain` (luôn từ chối), `baseline` (TF-IDF); XLM-R `xlmr` (seed 42), `xlmr_seed13`, `xlmr_256`
(cửa sổ 256/96); PhoBERT `phobert` (lr 3e-5, seed 42), `phobert_seed13`, `phobert_lr2e5`, `phobert_stable`
(lr 1e-5). Hai seed được báo cáo ngang nhau.
