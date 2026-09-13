"""
Ứng dụng Web Demo Hỏi Đáp Trích Xuất Tiếng Việt (Vietnamese Extractive MRC)
Môn học: CS221 - Xử Lý Ngôn Ngữ Tự Nhiên (UIT)
"""
import streamlit as st

st.set_page_config(
    page_title="Hệ Thống Đọc Hiểu & Hỏi Đáp Tiếng Việt (Vietnamese MRC)",
    page_icon="🇻🇳",
    layout="wide"
)

st.title("🇻🇳 Hệ Thống Đọc Hiểu & Trả Lời Câu Hỏi Tiếng Việt (Vietnamese Extractive MRC)")
st.caption("Đồ án môn học CS221 - Xử Lý Ngôn Ngữ Tự Nhiên | Giảng viên: ThS. Đặng Văn Thìn | Dataset: UIT-ViQuAD")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📝 Ngữ Cảnh Văn Bản (Context)")
    sample_context = (
        "Trường Đại học Công nghệ Thông tin (Đại học Quốc gia Thành phố Hồ Chí Minh, "
        "viết tắt là UIT) là một trường đại học thành viên chuyên ngành công nghệ thông tin "
        "và truyền thông tại Việt Nam, được thành lập vào năm 2006. Trường có trụ sở chính "
        "tại Khu đô thị ĐHQG-HCM, thuộc thành phố Thủ Đức, Thành phố Hồ Chí Minh."
    )
    context = st.text_area("Nhập đoạn văn bản ngữ cảnh:", value=sample_context, height=180)
    
    st.subheader("❓ Câu Hỏi (Question)")
    sample_question = "Trường Đại học Công nghệ Thông tin được thành lập vào năm nào?"
    question = st.text_input("Nhập câu hỏi của bạn:", value=sample_question)
    
    model_choice = st.selectbox(
        "Chọn mô hình dự đoán:",
        ["vinai/phobert-base-v2", "FPTAI/videberta-base", "bert-base-multilingual-cased", "Baseline BM25"]
    )
    
    predict_btn = st.button("🚀 Trả Lời Ngay", type="primary")

with col_right:
    st.subheader("🎯 Kết Quả Trích Xuất Câu Trả Lời (Extracted Answer)")
    if predict_btn:
        # Mock logic demo hiển thị kết quả
        ans = "năm 2006"
        score = 0.965
        
        st.success(f"**Câu trả lời dự đoán:** {ans}")
        st.info(f"**Độ tự tin (Confidence Score):** {score*100:.2f}% | **Thời gian phản hồi:** 18ms")
        
        # Bôi vàng từ khóa trong context
        highlighted = context.replace("năm 2006", "<mark style='background-color: #fef08a; font-weight: bold;'>năm 2006</mark>")
        st.markdown("### 🔍 Vị trí câu trả lời trong đoạn văn:")
        st.markdown(f"<div style='padding: 16px; background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; color: #1e293b;'>{highlighted}</div>", unsafe_allow_html=True)
