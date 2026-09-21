# Bộ stress-test v2

Bộ này thay thế bộ v1 250 câu, nay đã được gỡ khỏi kho mã. Kiểm định v1 cho thấy nó không dùng được: chỉ 15/120 câu có đáp án chứa đáp án vàng trong context, cả bộ chỉ có 5 context, câu hỏi sinh bằng template, và một hệ thống "luôn từ chối" đạt 52 EM, bằng các mô hình. Lịch sử đầy đủ nằm trong git (commit trước khi gỡ).

## Nguyên tắc

1. **Không tự viết câu hỏi, không tự gán nhãn.** Mọi câu hỏi và đáp án vàng lấy từ UIT-ViQuAD 2.0 validation, nhãn do người gán. Có hai loại mục:
   - **slice**: câu gốc, giữ nguyên văn, được chọn vì mang một hiện tượng đo được;
   - **perturbation**: câu gốc biến đổi theo một luật mà nhãn mới suy ra được từ nhãn cũ. Mỗi câu biến đổi đi kèm câu gốc (`role: "original"`) để đo độ nhất quán theo cặp.
2. **Mọi câu đều chấm được.** `scripts/build_stress_v2.py` kiểm từng câu: đáp án có trong context, offset đúng, câu bị xoá thật sự mất đáp án, câu nhiễu không chứa đáp án, context không có trong train. Script dừng nếu có vi phạm.
3. **Chỉ dùng để chẩn đoán.** Không huấn luyện, không chọn epoch, seed hay τ trên bộ này. Context lấy từ validation (0/557 context trùng train).

## Thành phần

Số liệu đầy đủ nằm ở `results/stress_v2_audit.json`.

| Mã | Hiện tượng | Tập con | Cách dựng | Số câu thử | Câu gốc đi cặp |
|---|---|---|---|---:|---:|
| E1 | Ranh giới từ ghép | E1a | slice: biên đáp án vàng cắt ngang một từ của pyvi (lấy **tất cả**) | 33 | – |
| | | E1b | slice: đáp án mở đầu hoặc kết thúc bằng từ ghép riêng viết hoa ≥2 âm tiết | 100 | – |
| E2 | Nhiễu / ngữ cảnh dài | E2a | slice: đáp án nằm sau âm tiết thứ 180 (ngoài cửa sổ đầu) | 80 | – |
| | | E2b | slice: đáp án chứa số, context có ≥3 con số khác | 80 | – |
| | | E2c | perturbation: chèn một câu nhiễu kiểu AddSent, chủ thể khác cùng loại, số khác | 63 | 63 |
| E3 | Không trả lời được | E3a | slice: câu impossible do người viết trong ViQuAD 2.0 | 100 | – |
| | | E3b | perturbation: xoá câu chứa đáp án khỏi context | 100 | 100 |
| E4 | Lệch từ vựng / nhiều câu | E4 | slice: câu chứa đáp án không phải câu trùng từ nhiều nhất với câu hỏi | 120 | – |
| E5 | Câu hỏi không dấu | E5 | perturbation: bỏ toàn bộ dấu tiếng Việt trong câu hỏi, context giữ nguyên | 150 | 150 |

**Tổng: 1.139 câu** (826 câu thử + 313 câu gốc), trong đó 839 câu có đáp án và 300 câu không; 572 context khác nhau. Mỗi tập con lấy tối đa 2 câu từ cùng một context.

E5 thay cho nhóm "nhãn mập mờ" của v1. Nhóm đó không dựng tự động được mà không tự gán nhãn. Còn gõ không dấu là hiện tượng có thật khi người dùng Việt đặt câu hỏi.

## Đọc kết quả

`scripts/run_eval.py --dataset stress2` ghi `by_category` và `by_subset`:

- Điểm nhóm **không tính câu gốc**, vì câu gốc là đối chứng chứ không phải phép thử.
- Với các tập con có cặp (E2c, E3b, E5), con số chính là `pairs.broken`: số câu mô hình **đúng ở câu gốc nhưng sai sau biến đổi**. Đây là lỗi do chính phép biến đổi gây ra, tách khỏi lỗi đọc hiểu vốn có.
- Báo cáo dùng **tỉ lệ hỏng** `broken / (both_correct + broken)`: trong số cặp mô hình vốn trả lời đúng câu gốc, bao nhiêu hỏng sau biến đổi. `both_correct` (nhất quán) là phần bù của nó, không phải một chỉ số khác — hai cách đọc cùng một phép đo.
- **E3** thì hệ "luôn từ chối" đạt 100 EM trên câu thử theo định nghĩa, nên không đọc E3 bằng EM. Nó cũng gần như không trả lời đúng câu gốc nào, nên mẫu số của tỉ lệ hỏng bằng 0: tỉ lệ đó không xác định cho "luôn từ chối" và TF-IDF, và các hệ này không có mặt trong bảng cặp.
- `run_eval.py` chấm ở τ = 0. Kết quả trong báo cáo ở **τ chọn trên dev**: chấm từng cửa sổ bằng `scripts/score_windows.py --split stress2`, rồi áp τ của `results/thresholds.json` bằng `scripts/score_stress2_tuned.py` → `results/eval_<run>_stress2_tuned.json`. τ không bao giờ được chọn trên bộ này.

## Kiểm tra sơ bộ trên dự đoán có sẵn

Các tập con slice là câu validation nguyên văn, nên chấm được ngay từ dự đoán validation đã có, bằng lệnh `python scripts/stress_v2_slice_scores.py`, kết quả ở `results/stress_v2_slice_scores.json`. Kết quả sơ bộ, EM ở τ dev:

- **E1a** (33 câu) tách được hai kiến trúc: PhoBERT 27,3 so với XLM-R 48,5–54,5. Điều này khớp với trần EM do tách từ, nhưng n nhỏ, CI rộng.
- **E4** khó hơn rõ rệt ở mọi mô hình (≈32–40 EM, so với ≈58 trên toàn bộ câu có đáp án).
- **E2a/E2b** *không* khó hơn trung bình. Ngữ cảnh dài và con số nhiễu tự nhiên không làm hai mô hình sai nhiều hơn. Phép thử nhiễu thật sự là E2c, cần chạy mô hình.

## Hạn chế cần nêu trong báo cáo

- **E2c và E3b dựng bằng luật, chưa kiểm tra tay.** Câu nhiễu có thể ngượng về ngữ nghĩa, và phần context còn lại sau khi xoá có thể vẫn diễn đạt lại đáp án. Vì vậy cả 163 câu nằm trong `review_sheet.csv`. Hai người điền cột `nguoi_1` và `nguoi_2` bằng `ok` hoặc `loai`, rồi chạy lại `scripts/build_stress_v2.py`: câu bị loại (cùng câu gốc) được bỏ, còn tỉ lệ đồng thuận và κ được ghi vào audit.
- **E2c chỉ có 63 câu** từ 8 bài. Luật yêu cầu tên bài là địa danh hoặc tên người, xuất hiện trong cả câu hỏi và câu chứa đáp án, và đáp án có chữ số.
- **Validation chỉ có 19 bài.** Mọi tập con trải trên nhiều context nhưng ít chủ đề.
- E1a phụ thuộc bộ tách từ pyvi, không phải RDRSegmenter mà PhoBERT dùng khi pretrain.
- E4 là heuristic trùng từ, không phân biệt suy luận nhiều bước thật với diễn đạt lại.

## Tái lập

```bash
python scripts/build_stress_v2.py              # cần pyvi; seed 42; dừng nếu có vi phạm
python -m pytest tests/test_stress_v2.py
python scripts/stress_v2_slice_scores.py       # không cần mô hình
python scripts/run_eval.py --models abstain baseline phobert_seed13 xlmr_seed13 xlmr_256 --dataset stress2
```
