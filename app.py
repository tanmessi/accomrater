# app.py

import streamlit as st
import torch
import os
import json
import numpy as np
import yaml
import pandas as pd
import time
from typing import List, Dict, Any

# Import components
from components.analyzer import get_sentiment_analyzer, analyze_with_gnn, analyze_multiple_reviews
from components.visualization import display_analysis_results, display_text_processing, display_summary_results
from components.crawler import show_crawler_section
from components.statistics import show_statistics_section
from components.data_processor import preprocess_text, create_node_features, get_text_preprocessor

# Import configs
from config.constants import (
    ASPECT_KEYWORDS_PATH, 
    EMOTION_WORDS_PATH, 
    CLASS_MAPPING, 
    CLASS_COLORS,
    AVAILABLE_MODELS,
    MODELS_DIR,
    MODEL_INFO
)

# Thiết lập tiêu đề và chế độ layout
st.set_page_config(
    page_title="Phân tích cảm xúc review khách sạn",
    page_icon="🏨",
    layout="wide"
)

# Hàm main để hiển thị giao diện ứng dụng
def main():
    st.title("🏨 Phân tích cảm xúc review khách sạn với GNN")
    
    # Tạo tabs cho các chức năng khác nhau
    tab1, tab2, tab3 = st.tabs(["Phân tích Review", "Thu thập dữ liệu", "Thống kê & Báo cáo"])
    
    with tab1:
        # Sidebar cho việc chọn model và hiển thị thông tin
        st.sidebar.header("Cài đặt")
        
        # Hiển thị tên model theo định dạng đơn giản
        model_display_name = lambda x: x.replace("best_", "").replace("_embedding.pt", "").replace(".pt", "")
        
        selected_model = st.sidebar.selectbox(
            "Chọn mô hình GNN",
            AVAILABLE_MODELS,
            format_func=model_display_name
        )
        
        # Xác định loại embedding từ tên model
        if "phobert" in selected_model:
            embedding_type = "phobert_embedding"
        elif "word2vec" in selected_model:
            embedding_type = "word2vec_embedding"
        elif "tfidf" in selected_model:
            embedding_type = "tfidf_embedding"
        else:
            st.sidebar.warning(f"Không thể xác định loại embedding từ tên model: {selected_model}")
            st.sidebar.info("Đang sử dụng loại embedding mặc định: phobert_embedding")
            embedding_type = "phobert_embedding"
        
        # Xác định loại model
        if "gcn" in selected_model:
            model_type = "gcn"
        elif "gat" in selected_model:
            model_type = "gat"
        elif "sage" in selected_model:
            model_type = "sage"
        else:
            model_type = "unknownmodel"
        
        st.sidebar.markdown(f"**Loại mô hình:** {model_type.upper()}")
        st.sidebar.markdown(f"**Loại embedding:** {embedding_type}")
        
        # Hiển thị thông tin về model
        # model_key = f"{model_type}_{embedding_type}"
        # if model_key in MODEL_INFO:
        #     st.sidebar.markdown("### Thông tin model")
        #     st.sidebar.markdown(f"**Accuracy:** {MODEL_INFO[model_key]['accuracy']:.3f}")
        #     st.sidebar.markdown(f"**F1 Score:** {MODEL_INFO[model_key]['f1']:.3f}")
        #     st.sidebar.markdown(f"**Training Time:** {MODEL_INFO[model_key]['time']}")
        
        # Hiển thị thông tin về quá trình phân tích
        st.sidebar.markdown("### Thông tin quá trình phân tích")
        show_processing = st.sidebar.checkbox("Hiển thị quá trình xử lý văn bản", value=False)
        
        # Hướng dẫn sử dụng
        # with st.sidebar.expander("Hướng dẫn sử dụng"):
        #     st.markdown("""
        #     ### Cách sử dụng
            
        #     1. **Chọn mô hình GNN** ở dropdown menu trên sidebar
            
        #     2. **Chọn chế độ nhập reviews**:
        #        - *Nhập một review*: Phân tích một review duy nhất
        #        - *Nhập nhiều reviews*: Phân tích hàng loạt reviews
            
        #     3. **Khi phân tích nhiều reviews**:
        #        - Nhập mỗi review trên một dòng, hoặc
        #        - Tải lên file CSV chứa reviews (bạn có thể tải về template)
               
        #     4. **Nhấn nút "Phân tích"** để xem kết quả
            
        #     5. **Xem kết quả phân tích**:
        #        - Tổng quan phân phối cảm xúc
        #        - Bảng chi tiết kết quả
        #        - Các khía cạnh được đề cập nhiều nhất
        #        - Chi tiết từng review
        #     """)
        
        # Giải thích về ý nghĩa cảm xúc
        # with st.sidebar.expander("Ý nghĩa điểm cảm xúc"):
        #     st.markdown("""
        #     ### Ý nghĩa điểm cảm xúc
            
        #     - **0.0 - 0.4**: Tiêu cực 🔴
        #     - **0.4 - 0.7**: Trung bình 🟠
        #     - **0.7 - 1.0**: Tích cực 🟢
            
        #     Điểm cảm xúc được tính dựa trên:
        #     - Phân tích từ khóa tích cực/tiêu cực trong review
        #     - Tổng hợp các khía cạnh được đề cập
        #     - Cường độ cảm xúc thể hiện trong review
        #     """)
        
        # Chọn chế độ nhập review
        input_mode = st.radio(
            "Chế độ nhập review",
            ["Nhập một review", "Nhập nhiều reviews"]
        )
        
        if input_mode == "Nhập một review":
            # Phần nhập văn bản đơn
            st.subheader("📝 Nhập review khách sạn cần phân tích")
            review_text = st.text_area("Nhập nội dung review:", height=150)
            
            # Nút phân tích
            if st.button("🔍 Phân tích review"):
                if not review_text:
                    st.warning("Vui lòng nhập nội dung review!")
                else:
                    with st.spinner("Đang xử lý và phân tích..."):
                        # Phân tích với model đã chọn
                        predicted_class, aspect_results, processed_text, overall_score, conclusions = analyze_with_gnn(
                            review_text, 
                            os.path.join(MODELS_DIR, selected_model),
                            embedding_type
                        )
                        
                        # Hiển thị kết quả nếu có
                        if predicted_class is not None:
                             # Hiển thị quá trình xử lý văn bản nếu được chọn
                            if show_processing:
                                display_text_processing(processed_text)
                            # Hiển thị thông tin về model
                            st.markdown("### Kết quả phân tích")
                            
                            # Xác định model_type và embedding_type từ selected_model
                            model_type = "GCN" if "gcn" in selected_model else "GAT" if "gat" in selected_model else "GraphSAGE" if "sage" in selected_model else "Unknown"
                            embedding_name = "PhoBERT" if "phobert" in selected_model else "Word2Vec" if "word2vec" in selected_model else "TF-IDF" if "tfidf" in selected_model else "Unknown"
                            
                            # Hiển thị thông tin model
                            st.info(f"Kết quả từ model **{model_type}** với embedding **{embedding_name}**")
                            
                            # Hiển thị kết quả phân tích
                            display_analysis_results(review_text, predicted_class, aspect_results, overall_score, conclusions)

                            
                            # # Hiển thị thông tin về embedding
                            # with st.expander("Thông tin kỹ thuật về embedding"):
                            #     st.markdown(f"**Loại embedding:** {embedding_type}")
                                
                            #     # Mô tả về phương pháp embedding
                            #     if embedding_type == "phobert_embedding":
                            #         st.markdown("""
                            #         **PhoBERT Embedding:**
                            #         - Sử dụng mô hình ngôn ngữ PhoBERT được huấn luyện đặc biệt cho tiếng Việt
                            #         - Tạo vector đặc trưng có độ dài 768 chiều
                            #         - Tính toán trung bình của các token embedding từ lớp cuối cùng
                            #         - Phù hợp đặc biệt cho phân tích ngữ nghĩa tiếng Việt
                            #        """)
                            #     elif embedding_type == "word2vec_embedding":
                            #         st.markdown("""
                            #         **Word2Vec Embedding:**
                            #         - Sử dụng phương pháp tạo vector từ dựa trên ngữ cảnh
                            #         - Tạo vector đặc trưng có độ dài 100 chiều
                            #         - Tính toán trung bình của các word vector trong văn bản
                            #         - Hiệu quả cho các tác vụ phân tích cấp từ và phân loại văn bản
                            #         """)
                            #     elif embedding_type == "tfidf_embedding":
                            #         st.markdown("""
                            #         **TF-IDF Embedding:**
                            #         - Sử dụng phương pháp tần suất từ - nghịch đảo tần suất văn bản
                            #         - Tạo vector đặc trưng có độ dài lên đến 1000 chiều
                            #         - Thể hiện độ quan trọng của từng từ trong văn bản
                            #         - Phù hợp cho phân loại văn bản và truy vấn thông tin
                            #         """)
                        else:
                            st.error("Không thể phân tích. Vui lòng thử lại!")
        else:
            # Phần nhập nhiều văn bản
            st.subheader("📝 Nhập nhiều reviews khách sạn cần phân tích")
            
            # Option 1: Nhập text trực tiếp
            reviews_text = st.text_area(
                "Nhập mỗi review trên một dòng:",
                height=200,
                placeholder="Review 1\nReview 2\nReview 3\n..."
            )
            
            # Option 2: Tải lên file CSV và tải về template
            col1, col2 = st.columns([2, 1])
            with col1:
                upload_file = st.file_uploader("Hoặc tải lên file CSV chứa reviews:", type=['csv'])
                
            with col2:
                # Tạo CSV template để download
                template_df = pd.DataFrame({
                    'review': ['Nhập review của bạn tại đây...', 'Khách sạn này rất tốt, nhân viên thân thiện', 'Phòng bẩn, nhân viên không nhiệt tình'],
                    'hotel_id': ['1', '2', '3'],
                    'rating': ['5', '4', '2']
                })
                
                # Convert DataFrame to CSV
                csv = template_df.to_csv(index=False)
                
                # Add download button
                st.download_button(
                    label="📥 Tải về CSV Template",
                    data=csv,
                    file_name="reviews_template.csv",
                    mime="text/csv",
                    help="Tải về file mẫu CSV để điền reviews và tải lên"
                )
            
            # Nút phân tích
            analyze_button = st.button("🔍 Phân tích tất cả reviews", key="analyze_button")
            
            if analyze_button:
                reviews_to_analyze = []
                
                # Xử lý reviews từ text area
                if reviews_text:
                    reviews_to_analyze.extend([r.strip() for r in reviews_text.split('\n') if r.strip()])
                
                # Xử lý reviews từ file CSV nếu có
                if upload_file is not None:
                    try:
                        df = pd.read_csv(upload_file)
                        # Hiển thị preview của dữ liệu CSV
                        with st.expander("Xem trước dữ liệu CSV đã tải lên"):
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Tìm cột có reviews (có thể là 'review', 'comment', 'text', etc.)
                        review_column = None
                        possible_columns = ['review', 'comment', 'text', 'content', 'description', 'feedback']
                        
                        for col in possible_columns:
                            if col in df.columns:
                                review_column = col
                                break
                        
                        # Nếu không tìm thấy, cho phép người dùng chọn cột
                        if review_column is None and len(df.columns) > 0:
                            review_column = st.selectbox(
                                "Vui lòng chọn cột chứa nội dung review:",
                                options=df.columns.tolist(),
                                key="column_selector"
                            )
                        
                        # Tìm cột hotel_id nếu có
                        hotel_id_column = None
                        possible_hotel_columns = ['hotel_id', 'hotel', 'hotel_name', 'accommodation_id']
                        
                        for col in possible_hotel_columns:
                            if col in df.columns:
                                hotel_id_column = col
                                break
                        
                        # Lưu DataFrame ban đầu vào session_state để có thể lọc mà không phải tải lại
                        if 'full_df' not in st.session_state:
                            st.session_state.full_df = df.copy()
                        
                        # Nếu có cột hotel_id, cho phép người dùng chọn hotel cụ thể
                        if hotel_id_column:
                            # Lấy danh sách các hotel_id duy nhất
                            unique_hotels = sorted(st.session_state.full_df[hotel_id_column].dropna().unique().tolist())
                            
                            # Thêm tùy chọn "Tất cả các khách sạn"
                            hotel_options = ["Tất cả các khách sạn"] + unique_hotels
                            
                            # Khởi tạo session_state cho hotel_selector nếu chưa có
                            if 'hotel_selection' not in st.session_state:
                                st.session_state.hotel_selection = "Tất cả các khách sạn"
                            
                            # Hàm xử lý khi thay đổi khách sạn
                            def update_hotel_selection():
                                st.session_state.hotel_selection = st.session_state.hotel_selector
                            
                            # Hiển thị dropdown chọn khách sạn
                            selected_hotel = st.selectbox(
                                f"Chọn khách sạn để phân tích (từ cột {hotel_id_column}):",
                                options=hotel_options,
                                index=hotel_options.index(st.session_state.hotel_selection),
                                key="hotel_selector",
                                on_change=update_hotel_selection
                            )
                            
                            # Lọc DataFrame theo khách sạn đã chọn
                            if st.session_state.hotel_selection != "Tất cả các khách sạn":
                                filtered_df = st.session_state.full_df[
                                    st.session_state.full_df[hotel_id_column] == st.session_state.hotel_selection
                                ]
                                df = filtered_df
                                st.info(f"Đang phân tích reviews cho khách sạn: {st.session_state.hotel_selection}")
                                st.write(f"Số lượng reviews: {len(filtered_df)}")
                            else:
                                df = st.session_state.full_df
                        
                        if review_column:
                            csv_reviews = df[review_column].dropna().tolist()
                            reviews_to_analyze.extend([str(r).strip() for r in csv_reviews if str(r).strip()])
                            st.success(f"Đã tải {len(csv_reviews)} reviews từ cột '{review_column}' của file CSV.")
                        else:
                            st.error("Không tìm được cột chứa reviews trong file CSV.")
                    except Exception as e:
                        st.error(f"Lỗi khi đọc file CSV: {str(e)}")
                
                # Kiểm tra số lượng reviews
                if not reviews_to_analyze:
                    st.warning("Không có reviews nào để phân tích. Vui lòng nhập nội dung review hoặc tải lên file CSV.")
                else:
                    # Giới hạn số lượng reviews để tránh quá tải
                    if len(reviews_to_analyze) > 100:
                        st.warning(f"Có quá nhiều reviews ({len(reviews_to_analyze)}). Chỉ phân tích 100 reviews đầu tiên.")
                        reviews_to_analyze = reviews_to_analyze[:100]
                    
                    # Phân tích tất cả reviews
                    analysis_results = analyze_multiple_reviews(
                        reviews_to_analyze,
                        os.path.join(MODELS_DIR, selected_model),
                        embedding_type
                    )
                    
                    # Hiển thị kết quả tổng hợp
                    if analysis_results:
                        st.success(f"Đã phân tích thành công {len(analysis_results)} reviews!")
                        display_summary_results(analysis_results)
                        
                        # Hiển thị chi tiết từng review
                        st.subheader("Chi tiết từng review")
                        for i, result in enumerate(analysis_results):
                            with st.expander(f"Review #{result['id']}: {result['review'][:100]}{'...' if len(result['review']) > 100 else ''}"):
                                st.markdown(f"**Review đầy đủ:**")
                                st.text(result['review'])
                                st.markdown("---")
                                
                                # Hiển thị kết quả phân tích cho review này
                                display_analysis_results(
                                    result['review'], 
                                    result['predicted_class'], 
                                    result['aspect_results'],
                                    result['overall_score'],
                                    result['conclusions'],
                                    key_suffix=f"_{i}"
                                )
                                
                                # Hiển thị quá trình xử lý văn bản nếu được chọn
                                if show_processing:
                                    st.markdown("#### Thông tin xử lý văn bản")
                                    st.text_area(
                                        "Văn bản sau khi xử lý:",
                                        result['processed_text']['final'],
                                        height=100,
                                        disabled=True,
                                        key=f"processed_text_{i}"
                                    )
                    else:
                        st.error("Không thể phân tích các reviews. Vui lòng thử lại!")
    
    # Tab 2: Thu thập dữ liệu
    with tab2:
        show_crawler_section()
    
    # Tab 3: Thống kê & Báo cáo
    with tab3:
        show_statistics_section()

if __name__ == "__main__":
    main()