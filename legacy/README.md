# legacy/ — khung mã ban đầu (không dùng để sinh kết quả)

Mã trong thư mục này là khung ban đầu của dự án. **Chưa từng được chạy** để sinh
bất kỳ con số nào trong báo cáo. Được giữ lại vì báo cáo (Mục kiểm định) phân tích
các lỗi của nó:

- `train_mrc_transformer.py`: `compute_metrics` từng so sánh chuỗi `span_{s}_{e}`
  thay vì văn bản đáp án; từng chứa bảng `EXPECTED_RESULTS` viết tay.
- `dataset_loader.py:624` — `include_stress_test=True` mặc định, trộn 60% bộ
  stress-test vào tập HUẤN LUYỆN (rò rỉ dữ liệu).
- Cấu hình PhoBERT ghi đè `max_position_embeddings`, làm hỏng embedding vị trí.

Pipeline dùng thật nằm ở `src/mrc/` (kế thừa từ đồ án CS116 của cùng nhóm, có
kiểm thử) và `scripts/`.
