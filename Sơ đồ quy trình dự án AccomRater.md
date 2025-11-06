# 📊 Sơ đồ quy trình dự án AccomRater 🔄

```mermaid
flowchart TD
    subgraph A["🔄 Thu thập dữ liệu (Data Collection)"]
        A1[booking_crawler.py]
        A2[agoda_crawler.py]
    end
    
    subgraph B["💾 Lưu trữ dữ liệu (Database)"]
        B1[(PostgreSQL)]
        B2[hotels]
        B3[reviews]
        B4[hotel_ratings]
        B5[sentiment_analysis]
        B1 --> B2
        B1 --> B3
        B1 --> B4
        B1 --> B5
    end
    
    subgraph C["🧹 Tiền xử lý dữ liệu (Preprocessing)"]
        C1[text_cleaning]
        C2[vietnamese_utils]
        C3[vectorization]
        C4[data_augmentation]
    end
    
    subgraph D["🕸️ Xây dựng đồ thị (Graph Building)"]
        D1[graph_builder.py]
        D2[node_features.py]
        D3[edge_features.py]
    end
    
    subgraph E["🧠 Mô hình GNN (GNN Model)"]
        E1[gcn_model.py]
        E2[gat_model.py]
        E3[trainer.py]
        E4[evaluation.py]
    end
    
    subgraph F["💭 Phân tích sentiment (Sentiment Analysis)"]
        F1[aspect_extractor.py]
        F2[sentiment_analyzer.py]
        F3[aspect_sentiment.py]
    end
    
    subgraph G["💡 Hệ thống gợi ý (Recommendation)"]
        G1[weakness_analyzer.py]
        G2[recommendation_generator.py]
        G3[knowledge_base.py]
    end
    
    subgraph H["📱 Giao diện người dùng (User Interface)"]
        H1[app.py]
        H2[components/]
        H3[screens/]
    end
    
    A -->|Dữ liệu thô| B
    B -->|Truy vấn dữ liệu| C
    C -->|Dữ liệu đã xử lý| D
    D -->|Đồ thị| E
    B -->|Đánh giá| F
    E -->|Mô hình huấn luyện| F
    F -->|Kết quả phân tích| G
    G -->|Đề xuất| H
    B -->|Thông tin khách sạn| H
```

## 🔄 Luồng hoạt động
1. **Thu thập dữ liệu**:
   - Crawlers thu thập thông tin từ Booking.com và Agoda.com
   - Xử lý phân trang, lazy loading, đánh giá tiếng Việt

2. **Lưu trữ dữ liệu**:
   - PostgreSQL database lưu thông tin khách sạn và đánh giá
   - Hệ thống hash để tránh trùng lặp dữ liệu

3. **Tiền xử lý dữ liệu**:
   - Làm sạch văn bản (HTML, emoji, URL)
   - Xử lý tiếng Việt (tách từ, sửa lỗi chính tả, xử lý teencode)
   - Vectorization (TF-IDF, Word2Vec, PhoBERT)

4. **Xây dựng đồ thị**:
   - Tạo nodes: khách sạn, người dùng, đánh giá, khía cạnh dịch vụ
   - Xây dựng edges thể hiện mối quan hệ
   - Trích xuất đặc trưng cho nodes và edges

5. **Mô hình GNN**:
   - Triển khai GCN và GAT
   - Huấn luyện với batch gradient descent
   - Đánh giá hiệu năng với cross-validation

6. **Phân tích sentiment**:
   - Trích xuất khía cạnh từ đánh giá
   - Phân tích sentiment cho từng khía cạnh
   - Tổng hợp điểm sentiment theo nhiều tiêu chí

7. **Hệ thống gợi ý**:
   - Xác định điểm yếu dựa trên so sánh với benchmark
   - Phân tích tần suất từ khóa tiêu cực
   - Đề xuất cải thiện dựa trên knowledge base

8. **Giao diện người dùng**:
   - Dashboard Streamlit hiển thị phân tích
   - Tương tác và filter theo nhu cầu
   - Đề xuất cải thiện cụ thể cho chủ khách sạn

## 🔄 Triển khai

```
📂 AccomRater/
├── 📄 app.py                         # Ứng dụng Streamlit chính
├── 📂 crawlers/                      # Thu thập dữ liệu
│   ├── 📄 booking_crawler.py
│   └── 📄 agoda_crawler.py
├── 📂 data_preprocessing/            # Xử lý dữ liệu
│   ├── 📂 text_cleaning/
│   ├── 📂 vietnamese_utils/
│   └── 📂 vectorization/
├── 📂 model_training/                # Huấn luyện mô hình
│   ├── 📂 graph/
│   └── 📂 sentiment/
├── 📂 recommendation/                # Hệ thống gợi ý
│   ├── 📄 weakness_analyzer.py
│   └── 📄 recommendation_generator.py
└── 📂 ui/                            # Giao diện người dùng
    └── 📂 components/
```

📍 Triển khai: Docker + Docker Compose
🔄 Trạng thái: Hoàn thành v1.0.0