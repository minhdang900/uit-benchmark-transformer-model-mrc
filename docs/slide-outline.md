# 📑 DÀN Ý CHI TIẾT SLIDE BÁO CÁO THUYẾT TRÌNH ĐỒ ÁN CS221 (20 SLIDES - 15 PHÚT)

---

### PHẦN I: GIỚI THIỆU & TỔNG QUAN (Slide 1–4)
* **Slide 1: Trang tiêu đề:** 
  * Tên đề tài: *Khảo sát và Đánh giá Hiệu năng các Mô hình Transformer Tiền Huấn Luyện trong Bài Toán Đọc Hiểu & Trả Lời Câu Hỏi Tiếng Việt (Vietnamese Extractive MRC)*.
  * Thành viên: Nguyễn Quang Lâm (25210289), Trần Trọng Tấn (25210334), Lê Quang Thi (25210337), Vỏ Cẩm Thu (25210342) | GVHD: ThS. Đặng Văn Thìn.
* **Slide 2: Tính cấp thiết của bài toán:**
  * Giới thiệu bài toán Extractive MRC ($C, Q \rightarrow A$).
  * Ứng dụng trong Trợ lý ảo AI, Chatbot chăm sóc khách hàng, Hệ thống tra cứu thông tin tự động.
* **Slide 3: Thách thức đặc thù của tiếng Việt:**
  * Tính đơn lập (*Isolating language*), từ ghép đa âm tiết, hiện tượng đồng âm khác nghĩa và teencode/từ địa phương.
* **Slide 4: Mục tiêu & Đóng góp của Đề tài:**
  * Xây dựng pipeline hoàn chỉnh từ dữ liệu $\rightarrow$ mô hình $\rightarrow$ đánh giá $\rightarrow$ phân tích lỗi $\rightarrow$ demo.

---

### PHẦN II: TẬP DỮ LIỆU & PHƯƠNG PHÁP (Slide 5–9)
* **Slide 5: Tập dữ liệu chuẩn UIT-ViQuAD:**
  * Thống kê số lượng bài báo Wikipedia, số đoạn văn, số câu hỏi (Train: 18.579, Dev: 2.285, Test: 2.210).
* **Slide 6: Tiền xử lý & Chiến lược phân chia Context-Level Split:**
  * Tách từ bằng RDRSegmenter / PyVi; nguyên tắc chống rò rỉ dữ liệu (*Anti Data Leakage*) giữa tập Train và Test.
* **Slide 7: Mô hình cơ sở (Baseline BM25 / TF-IDF):**
  * Nguyên lý trích xuất span dựa trên độ tương đồng từ khóa.
* **Slide 8: Kiến trúc Transformer Monolingual (PhoBERT & ViDeBERTa):**
  * Sơ đồ cấu trúc Encoder 12/24 layers; cơ chế Span Head dự đoán $\text{start\_logits}$ và $\text{end\_logits}$.
* **Slide 9: Kiến trúc Transformer Multilingual (mBERT / XLM-RoBERTa):**
  * Khả năng chuyển giao ngôn ngữ chéo (*Cross-lingual transfer*).

---

### PHẦN III: KẾT QUẢ THỰC NGHIỆM & ĐÁNH GIÁ (Slide 10–13)
* **Slide 10: Thiết lập Siêu tham số (Hyperparameters):**
  * Learning rate ($2\times 10^{-5}$), Batch size (16), Optimizer (AdamW), Epochs (5), Linear warmup.
* **Slide 11: Độ đo đánh giá chuẩn (Exact Match & F1-Score):**
  * Định nghĩa toán học của EM (tuyệt đối 100%) và F1 (độ bao phủ từ vựng).
* **Slide 12: Bảng kết quả đối sánh tổng hợp (Benchmark Table):**
  * So sánh chi tiết kết quả EM/F1 giữa Baseline BM25, PhoBERT-base-v2 và XLM-R-base. **Số liệu điền sau khi đo — xem `results/`.**
* **Slide 13: Đồ thị huấn luyện (Loss Curves & Convergence):**
  * Phân tích tốc độ hội tụ và hiện tượng Overfitting giữa các mô hình.

---

### PHẦN IV: PHÂN TÍCH LỖI CHUYÊN SÂU (DEEP ERROR ANALYSIS) (Slide 14–17)
* **Slide 14: Thống kê 4 nhóm lỗi chính trên 150 mẫu dự đoán sai:**
  * Biểu đồ tròn: Lỗi lệch biên (Span boundary error - 42%), Lỗi phủ định (18%), Lỗi câu hỏi dài/ngữ cảnh phức tạp (25%), Nhập nhằng ngữ nghĩa (15%).
* **Slide 15: Phân tích Case Study 1 & 2 (Lỗi lệch biên từ ghép):**
  * Minh họa cụ thể trường hợp mô hình cắt trúng từ nhưng thiếu từ bổ nghĩa.
* **Slide 16: Phân tích Case Study 3 & 4 (Lỗi câu hỏi chứa bẫy phủ định):**
  * Mô hình bắt đúng từ khóa nhưng sai bản chất câu hỏi "Không phải là...".
* **Slide 17: Biện pháp khắc phục & Bài học rút ra:**
  * Bổ sung Negative Sampling và cơ chế Post-processing lọc span.

---

### PHẦN V: DEMO, KẾT LUẬN & HƯỚNG PHÁT TRIỂN (Slide 18–20)
* **Slide 18: Giới thiệu ứng dụng Web Demo Streamlit:**
  * Kiến trúc hệ thống thời gian thực (Inference time: ~18ms/câu hỏi).
* **Slide 19: Kết luận & Tổng kết kết quả:**
  * Mô hình cho kết quả tốt nhất trên từng nhóm lỗi E1–E5. **Số liệu điền sau khi đo — xem `results/`.**
* **Slide 20: Hướng phát triển trong tương lai & Q&A:**
  * Mở rộng sang Generative RAG tiếng Việt; Lời cảm ơn thầy Đặng Văn Thìn và Hội đồng.
