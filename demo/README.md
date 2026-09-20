# ViMRC Console

Giao diện chẩn đoán cho đồ án: chạy mô hình thật trên một cặp câu hỏi – ngữ cảnh, và
đọc lại toàn bộ kết quả trong `results/` theo năm trang.

> **Không phải sản phẩm nộp.** Đồ án CS221 chỉ yêu cầu huấn luyện mô hình và báo cáo;
> console này là phần thêm để trình bày và soi lỗi. Báo cáo, slide và số liệu không phụ
> thuộc vào nó.

## Chạy

```bash
# streamlit KHÔNG nằm trong requirements.txt của pipeline chấm điểm
uv pip install --python .venv/bin/python -r demo/requirements.txt

# chạy TỪ GỐC REPO để .streamlit/config.toml và results/ được tìm thấy
.venv/bin/streamlit run demo/app.py
```

Mở <http://localhost:8501>.

## Năm trang

| Trang | Đọc từ | Nội dung |
|---|---|---|
| **Hỏi đáp** | `models/<run>/` + `data/raw/` | chạy checkpoint thật; đối chiếu đáp án với nhãn vàng, tô vị trí đáp án trong ngữ cảnh, so các nhánh mô hình trên cùng đầu vào |
| **Ma trận chẩn đoán** | `eval_*_validation.json`, `eval_*_stress.json`, `diagnosis_validation.json` | bảng kết quả chính, ma trận E1–E5, phân loại lỗi |
| **Khám phá lỗi** | `disagreements_phobert_xlmr.json` | các câu PhoBERT và XLM-R trả lời khác nhau, lọc theo cơ chế lỗi |
| **Bộ stress-test** | `stress_test_audit.json`, `data/stress_test/` | kiểm định từng nhóm, cảnh báo sinh từ chính số kiểm định, mẫu câu |
| **Huấn luyện & ngưỡng** | `training_curve_*.json`, `thresholds.json` | đường cong dev, các lần chạy, τ chọn trên dev |

## Quy ước

- **Không con số nào gõ tay.** Mọi giá trị đọc từ `results/*.json` qua `console/data.py`;
  thiếu tệp thì hiện dấu `—`, không đoán.
- **Trang Hỏi đáp dùng đúng lớp inference của báo cáo** (`mrc.transformer_qa.TransformerQA`)
  và đúng τ trong `thresholds.json`, nên đáp án khớp với bảng kết quả. Nút `τ = 0` cho
  thấy hành vi ở ngưỡng mặc định — chênh lệch giữa hai chế độ chính là phát hiện trung
  tâm của đồ án.
- **Nhãn vàng chỉ hiện khi câu hỏi chưa bị sửa** khỏi câu mẫu; tự nhập câu mới thì không
  có nhãn để chấm.
- Cần checkpoint trong `models/` (không commit) cho trang Hỏi đáp; bốn trang còn lại chạy
  được mà không cần GPU.

## Hệ thiết kế

Token màu/chữ/khoảng cách lấy từ hệ thiết kế Organic trên claude.ai/design
(`console/theme.py`). Bản thiết kế dùng Be Vietnam Pro cho cả chữ tiêu đề và nội dung vì
Caprasimo/Figtree không có bộ ký tự tiếng Việt.

Streamlit tự sinh DOM nên `theme.py` ghi đè CSS cho nút, ô nhập và thanh bên; nội dung
dạng bảng / ô nhiệt / thanh xếp lớp được dựng thành HTML trong `console/pages.py`.

## Cấu trúc

```
demo/
├── app.py              khung trang: thanh bên, tiêu đề, điều hướng
├── console/
│   ├── data.py         mọi số liệu, đọc từ results/
│   ├── infer.py        nạp checkpoint, chạy dự đoán, chấm EM/F1
│   ├── theme.py        token hệ thiết kế + ghi đè Streamlit
│   └── pages.py        năm trang
└── requirements.txt    chỉ streamlit
```

## Ghi chú vận hành

`.streamlit/config.toml` tắt bộ theo dõi tệp: bộ này quét mọi module đã import để tìm
đường dẫn, và với `transformers` thì việc đó kích hoạt chuỗi import lười rồi đụng
`torchvision` (phụ thuộc tuỳ chọn mà đồ án không cài). Sửa mã thì tự tải lại trang.
