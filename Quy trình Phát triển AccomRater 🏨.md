# 📄 Wiki: Quy trình Phát triển AccomRater 🏨

## 📑 Mục lục

1. [Tổng quan dự án](#tổng-quan-dự-án)
2. [Thu thập dữ liệu](#thu-thập-dữ-liệu)
3. [Tiền xử lý dữ liệu](#tiền-xử-lý-dữ-liệu)
4. [Xây dựng đồ thị](#xây-dựng-đồ-thị)
5. [Mô hình GNN](#mô-hình-gnn)
6. [Phân tích sentiment](#phân-tích-sentiment)
7. [Hệ thống gợi ý](#hệ-thống-gợi-ý)
8. [Triển khai ứng dụng](#triển-khai-ứng-dụng)

## 🎯 Tổng quan dự án

AccomRater: Hệ thống phân tích và gợi ý cải thiện dịch vụ lưu trú.

✅ Mục tiêu:
↳ Thu thập đánh giá từ Booking.com và Agoda.com
↳ Phân tích cảm xúc theo khía cạnh dịch vụ
↳ Xây dựng mô hình GNN cho phân tích kết nối
↳ Gợi ý cải thiện dịch vụ dựa trên phân tích

📦 Phiên bản: v1.0.0

## 🕸️ Thu thập dữ liệu

### 🔄 Crawling từ Booking.com

```bash
python run_booking_crawler.py
```

✅ Thực hiện:
↳ Khởi tạo BookingCrawler với tham số từ .env
↳ Mở trang tìm kiếm khách sạn với Selenium
↳ Scroll trang để tải thêm kết quả
↳ Thu thập URLs của các khách sạn
↳ Truy cập từng URL để lấy thông tin chi tiết
↳ Lọc và thu thập đánh giá tiếng Việt

⚙️ Tham số cấu hình:
```
HEADLESS_MODE=true
CRAWL_DELAY=3
MAX_HOTELS=100
MAX_REVIEWS_PER_HOTEL=300
```

### 🔄 Crawling từ Agoda.com

```bash
python run_agoda_crawler.py
```

✅ Thực hiện:
↳ Sử dụng GraphQL API hoặc Selenium tùy theo cấu hình
↳ Thu thập thông tin cơ bản của khách sạn
↳ Truy cập trang chi tiết để lấy thông tin và đánh giá
↳ Tạo hash cho mỗi đánh giá để tránh trùng lặp

⚠️ Lưu ý:
→ Kiểm tra selectors thường xuyên vì website thay đổi
→ Thiết lập CRAWL_DELAY đủ lớn để tránh bị chặn
→ Sử dụng proxy rotation nếu cần thu thập dữ liệu lớn

### 💾 Lưu trữ dữ liệu

✅ Cấu trúc DB:
↳ PostgreSQL với schema đã định nghĩa
↳ 4 bảng chính: hotels, reviews, hotel_ratings, sentiment_analysis

🔄 Quy trình:
```python
# Lưu thông tin khách sạn
hotel = Hotel.create(
    name=hotel_data['name'],
    address=hotel_data['address'],
    rating=rating,
    source="Booking.com",
    hotel_url=url
)

# Lưu đánh giá
for review in reviews_data:
    Review.create(
        hotel_id=hotel.id,
        rating=review_rating,
        comment=review.get('comment', ''),
        review_date=review_date_obj,
        source="Booking.com",
        reviewer_name=review.get('reviewer_name'),
        reviewer_hash=review.get('reviewer_hash')
    )
```

📍 Vị trí: /crawlers/booking_crawler.py, /crawlers/agoda_crawler.py

## 🧹 Tiền xử lý dữ liệu

### 🔄 Làm sạch văn bản

```python
class TextPreprocessor:
    def preprocess_pipeline(self, text):
        # Cleaning
        text = self.clean_text(text)
        # Vietnamese processing
        text = self.correct_spelling(text)
        text = self.handle_teencode(text)
        text = self.segment_text(text)
        return text
```

✅ Thực hiện:
↳ Loại bỏ HTML tags
↳ Xử lý emoji (thay thế hoặc loại bỏ)
↳ Loại bỏ URL
↳ Chuẩn hóa whitespace

📍 Vị trí: /data_preprocessing/preprocessing.py

### 🇻🇳 Xử lý tiếng Việt

✅ Thực hiện:
↳ Sửa lỗi chính tả với SymSpellPy
↳ Xử lý từ địa phương (mapping từ điển)
↳ Xử lý teencode ("k" → "không", "vk" → "vợ")
↳ Tách từ (word segmentation) với PyVi

```python
# Ví dụ: Xử lý teencode và từ địa phương
class VietnameseLocalDictionary:
    def normalize_text(self, text):
        words = text.split()
        for i, word in enumerate(words):
            if word in self.local_dict:
                words[i] = self.local_dict[word]
            elif word in self.teencode_dict:
                words[i] = self.teencode_dict[word]
        return ' '.join(words)
```

📍 Vị trí: /data_preprocessing/vietnamese_utils/

### 📊 Trích xuất đặc trưng

✅ Thực hiện:
↳ TF-IDF Vectorization
↳ Word Embeddings với Word2Vec hoặc FastText
↳ PhoBERT Embeddings đặc biệt cho tiếng Việt

```python
def get_document_embedding(self, text):
    if self.model_type == 'phobert':
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return embeddings.numpy()
```

📍 Vị trí: /data_preprocessing/vectorization/

## 🕸️ Xây dựng đồ thị

### 🔄 Tạo đồ thị từ dữ liệu quan hệ

✅ Loại đỉnh (node):
↳ Khách sạn (hotel nodes)
↳ Người dùng (user nodes)
↳ Đánh giá (review nodes)
↳ Khía cạnh dịch vụ (aspect nodes)

✅ Loại cạnh (edge):
↳ User-Review: Người dùng viết đánh giá
↳ Review-Hotel: Đánh giá thuộc khách sạn
↳ Review-Aspect: Đánh giá đề cập đến khía cạnh
↳ Hotel-Aspect: Khách sạn có khía cạnh dịch vụ

```python
def build_graph(self):
    G = nx.Graph()
    
    # Thêm hotel nodes
    for hotel in self.hotels:
        G.add_node(f"hotel_{hotel.id}", type='hotel', data=hotel)
    
    # Thêm review nodes và kết nối
    for review in self.reviews:
        G.add_node(f"review_{review.id}", type='review', data=review)
        G.add_edge(f"review_{review.id}", f"hotel_{review.hotel_id}")
        
        # Nếu có reviewer_hash, thêm user node
        if review.reviewer_hash:
            G.add_node(f"user_{review.reviewer_hash}", type='user')
            G.add_edge(f"user_{review.reviewer_hash}", f"review_{review.id}")
            
    # Thêm aspect nodes và kết nối
    for sentiment in self.sentiments:
        G.add_node(f"aspect_{sentiment.aspect}", type='aspect')
        G.add_edge(f"review_{sentiment.review_id}", f"aspect_{sentiment.aspect}", 
                  weight=sentiment.weight, score=sentiment.sentiment_score)
    
    return G
```

📍 Vị trí: /model_training/graph/graph_builder.py

### 🧮 Tính toán đặc trưng cho đồ thị

✅ Đặc trưng node:
↳ Hotel nodes: rating, số lượng review, vị trí địa lý
↳ Review nodes: rating, độ dài review, sentiment score
↳ User nodes: số lượng review, thời gian hoạt động
↳ Aspect nodes: one-hot encoding cho từng khía cạnh

✅ Đặc trưng edge:
↳ Độ mạnh của sentiment (weight)
↳ Sentiment score
↳ Thời gian (temporal features)

📍 Vị trí: /model_training/graph/feature_extraction.py

## 🧠 Mô hình GNN

### 📐 Kiến trúc mô hình

✅ Mô hình GCN:
```python
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        return x
```

✅ Mô hình GAT:
```python
class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

📍 Vị trí: /model_training/graph/models.py

### 🔄 Huấn luyện mô hình

✅ Thiết lập:
↳ Batch size: 32
↳ Learning rate: 0.001
↳ Epochs: 100
↳ Loss function: Cross Entropy (phân loại) hoặc MSE (regression)
↳ Optimizer: Adam

```python
def train(self, model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    out = model(data.x, data.edge_index, data.edge_attr)
    # Calculate loss
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    # Backward pass
    loss.backward()
    optimizer.step()
    return loss.item()
```

⚠️ Lưu ý:
→ Sử dụng cross-validation để tránh overfitting
→ Lưu mô hình tốt nhất dựa trên val_loss
→ Kiểm soát early stopping

📍 Vị trí: /model_training/graph/train.py

## 💭 Phân tích sentiment

### 🔍 Trích xuất khía cạnh dịch vụ

✅ Khía cạnh chính:
↳ Phòng (room): sạch sẽ, tiện nghi, rộng rãi...
↳ Nhân viên (staff): thân thiện, chuyên nghiệp...
↳ Dịch vụ (service): chất lượng, đa dạng...
↳ Vị trí (location): gần trung tâm, an ninh...
↳ Giá cả (price): hợp lý, đắt, rẻ...
↳ Ẩm thực (food): ngon, đa dạng, phong phú...

✅ Trích xuất khía cạnh:
↳ Sử dụng mô hình GNN để phân loại text fragments
↳ Kết hợp với lexicon-based approach
↳ Áp dụng dependency parsing để xác định quan hệ

### 🔄 Phân tích sentiment theo khía cạnh

✅ Quy trình:
↳ Tách comment thành các cụm theo khía cạnh
↳ Xác định sentiment score cho mỗi cụm (-1 đến 1)
↳ Gán trọng số cho mỗi khía cạnh dựa trên độ dài và vị trí
↳ Tổng hợp sentiment scores theo khía cạnh

```python
def analyze_aspect_sentiment(self, review_text):
    aspects = {}
    # Phân đoạn văn bản theo khía cạnh
    segments = self.aspect_segmenter.segment(review_text)
    
    for segment in segments:
        aspect = segment['aspect']
        text = segment['text']
        # Phân tích sentiment
        score = self.sentiment_analyzer.analyze(text)
        weight = len(text) / len(review_text)
        
        if aspect in aspects:
            aspects[aspect]['score'] += score * weight
            aspects[aspect]['weight'] += weight
        else:
            aspects[aspect] = {
                'score': score * weight,
                'weight': weight,
                'keywords': self.extract_keywords(text)
            }
    
    # Chuẩn hóa scores
    for aspect in aspects:
        aspects[aspect]['score'] /= aspects[aspect]['weight']
    
    return aspects
```

📍 Vị trí: /sentiment_analysis/aspect_sentiment.py

## 💡 Hệ thống gợi ý

### 📊 Phân tích điểm yếu

✅ Thực hiện:
↳ Tổng hợp sentiment scores theo khía cạnh
↳ Xác định khía cạnh có điểm thấp nhất
↳ So sánh với benchmark của các khách sạn tương tự
↳ Xác định mức độ ưu tiên cải thiện

```python
def identify_weaknesses(self, hotel_id):
    # Lấy sentiment scores theo khía cạnh
    aspect_scores = self.get_aspect_scores(hotel_id)
    
    # So sánh với benchmark
    benchmarks = self.get_benchmarks(hotel_id)
    
    weaknesses = []
    for aspect, score in aspect_scores.items():
        if score['avg_score'] < benchmarks[aspect]:
            gap = benchmarks[aspect] - score['avg_score']
            weaknesses.append({
                'aspect': aspect,
                'score': score['avg_score'],
                'benchmark': benchmarks[aspect],
                'gap': gap,
                'priority': self.calculate_priority(gap, aspect)
            })
    
    # Sắp xếp theo mức độ ưu tiên
    return sorted(weaknesses, key=lambda x: x['priority'], reverse=True)
```

📍 Vị trí: /recommendation/weakness_analyzer.py

### 🚀 Đề xuất cải thiện

✅ Thực hiện:
↳ Phân tích từ khóa phổ biến trong đánh giá tiêu cực
↳ Tìm các pattern lặp lại trong các vấn đề
↳ Tổng hợp recommender knowledge base
↳ Đề xuất các hành động cụ thể theo vấn đề

```python
def generate_recommendations(self, hotel_id):
    # Xác định điểm yếu
    weaknesses = self.weakness_analyzer.identify_weaknesses(hotel_id)
    
    recommendations = []
    for weakness in weaknesses:
        aspect = weakness['aspect']
        # Trích xuất từ khóa tiêu cực
        negative_keywords = self.get_negative_keywords(hotel_id, aspect)
        
        # Tìm pattern phổ biến
        patterns = self.identify_patterns(negative_keywords)
        
        # Tạo đề xuất từ knowledge base
        aspect_recommendations = self.recommendation_knowledge.get_recommendations(
            aspect, patterns, weakness['score']
        )
        
        recommendations.append({
            'aspect': aspect,
            'score': weakness['score'],
            'gap': weakness['gap'],
            'keywords': negative_keywords,
            'recommendations': aspect_recommendations
        })
    
    return recommendations
```

⚙️ Ví dụ đề xuất:
- Phòng (3.2/5): 
  - Cải thiện cách âm giữa các phòng
  - Nâng cấp hệ thống điều hòa
  - Thay mới đồ vải giường
- Nhân viên (3.8/5):
  - Tổ chức training giao tiếp tiếng Anh
  - Cải thiện quy trình check-in/check-out

📍 Vị trí: /recommendation/recommendation_generator.py

## 🚀 Triển khai ứng dụng

### 📊 Dashboard và UI

✅ Thực hiện:
↳ Xây dựng dashboard với Streamlit
↳ Hiển thị trực quan các phân tích sentiment
↳ Cung cấp các đề xuất cải thiện
↳ Tích hợp filter và tùy chỉnh

```python
def show():
    st.title("AccomRater - Phân tích và Đề xuất")
    
    # Chọn khách sạn
    hotel_id = st.selectbox("Chọn khách sạn", options=get_hotel_list())
    
    if st.button("Phân tích"):
        # Phân tích sentiment
        sentiment_results = analyze_sentiment(hotel_id)
        
        # Hiển thị biểu đồ
        st.subheader("Phân tích sentiment theo khía cạnh")
        display_sentiment_chart(sentiment_results)
        
        # Hiển thị đề xuất
        st.subheader("Đề xuất cải thiện")
        recommendations = generate_recommendations(hotel_id)
        display_recommendations(recommendations)
```

📍 Vị trí: /ui/dashboard.py

### 🔄 Cập nhật dữ liệu tự động

✅ Thực hiện:
↳ Cron job chạy crawler định kỳ (hàng tuần)
↳ Cập nhật dữ liệu vào database
↳ Tái huấn luyện mô hình với dữ liệu mới
↳ Cập nhật đề xuất

```bash
# Ví dụ crontab
0 0 * * 0 cd /path/to/accomrater && python run_crawlers.py >> logs/cron.log 2>&1
0 2 * * 0 cd /path/to/accomrater && python retrain_models.py >> logs/cron.log 2>&1
```

⚠️ Lưu ý:
→ Xử lý lỗi và backup dữ liệu trước mỗi lần cập nhật
→ Lưu lại metrics để theo dõi hiệu suất mô hình theo thời gian
→ Thông báo khi phát hiện thay đổi đáng kể trong data distribution

## 📦 Tài liệu tham khảo

- [Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
- [Aspect-Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004.pdf)
- [PhoBERT: Pre-trained language models for Vietnamese](https://arxiv.org/abs/2003.00744)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

---

**🔄 Trạng thái:** Đã hoàn thành version 1.0
**📅 Cập nhật:** 04/03/2025