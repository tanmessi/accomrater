#!/bin/bash

# Tạo thư mục ml_models nếu chưa tồn tại
mkdir -p ml_models

# Cài đặt các dependency cần thiết
pip install streamlit torch transformers

# Khởi chạy ứng dụng Streamlit
echo "Khởi chạy ứng dụng Streamlit..."
streamlit run app.py