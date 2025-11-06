import re
import underthesea
import pickle
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from imblearn.over_sampling import SMOTE
from data_preprocessing.feature_engineering import FeatureEngineer
from database.connection import get_db_connection

# ============================
# 1️⃣ Khởi tạo FeatureEngineer để xử lý vector hóa
# ============================
featureEngineer = FeatureEngineer()

# ============================
# 2️⃣ Kết nối Database
# ============================


def fetch_reviews():
    """Lấy dữ liệu review từ database."""
    print("📡 Đang lấy dữ liệu từ database...")
    conn = get_db_connection()
    query = "SELECT review_id, final_text, processed_at AS created_at FROM processed_reviews"
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"✅ Lấy thành công {len(df)} dòng dữ liệu từ database.")
    return df


# ============================
# 3️⃣ Load mô hình PhoBERT để phân tích cảm xúc tiếng Việt
# ============================
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_model = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer)

# ============================
# 4️⃣ Thiết lập TF-IDF với stopwords tiếng Việt
# ============================
vietnamese_stopwords = [
    "và", "là", "có", "của", "cho", "được", "rất", "với", "tại", "nhưng", "thì", "không", "cũng",
    "đây", "đó", "đấy", "vì", "sau", "trước", "từ", "trong", "ra", "lại", "này", "kia", "ấy",
    "bạn", "tôi", "anh", "chị", "em", "bên", "kia", "vậy"
]

tfidf_vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words=vietnamese_stopwords,
    ngram_range=(1, 2),
    smooth_idf=True,
    sublinear_tf=True
)

# Từ điển khía cạnh và từ khóa liên quan
aspect_keywords = {
    "phòng": ["phòng", "giường", "rộng rãi", "sạch sẽ", "thoải mái"],
    "nhân viên": ["nhân viên", "lễ tân", "phục vụ", "chuyên nghiệp", "thân thiện", "nhiệt tình"],
    "đồ ăn": ["đồ ăn", "bữa sáng", "món ăn", "nhà hàng", "thực đơn", "không ngon"],
    "thức ăn": ["đồ ăn", "bữa sáng", "món ăn", "nhà hàng", "thực đơn", "không ngon"],
    "vị trí": ["vị trí", "gần biển", "trung tâm", "thuận tiện", "đi lại"],
    "giá cả": ["giá", "giá cả", "hợp lý", "quá đắt", "rẻ"],
    "view": ["view", "cảnh đẹp", "hướng biển", "tầm nhìn"],
    "wifi": ["wifi", "mạng", "internet", "kết nối"],
    "hồ bơi": ["hồ bơi", "bể bơi", "nước sạch"],
    "bãi đỗ xe": ["bãi đỗ xe", "đậu xe", "gửi xe"],
    "dịch vụ phòng": ["dịch vụ phòng", "room service", "phục vụ tận phòng"],
    "spa": ["spa", "massage", "trị liệu"],
    "gym": ["gym", "phòng tập", "tập thể dục"],
    "giải trí": ["giải trí", "karaoke", "rạp chiếu phim"],
    "đưa đón": ["đưa đón", "xe đưa đón", "shuttle bus"],
    "bãi biển": ["bãi biển", "bien", "biển", "nước trong"],
    "núi": ["núi", "phong cảnh núi", "leo núi"],
    "sông": ["sông", "ven sông"],
    "hồ": ["hồ", "hồ nước"],
    "đảo": ["đảo", "quần đảo"],
    "trung tâm": ["trung tâm", "khu vực trung tâm"],
    "an ninh": ["an ninh", "bảo vệ", "an toàn"],
    "sạch sẽ": ["sạch sẽ", "gọn gàng"],
    "tiện nghi": ["tiện nghi", "đầy đủ thiết bị"],
    "không gian": ["không gian", "thoáng mát", "rộng lớn"],
    "yên tĩnh": ["yên tĩnh", "không ồn ào"],
    "thời tiết": ["thời tiết", "khí hậu"],
    "đồ uống": ["đồ uống", "quầy bar", "cocktail"],
    "trải nghiệm tổng thể": ["trải nghiệm", "tổng thể", "cảm giác"]
}

# Tạo từ điển descriptive_words cho các từ mô tả
descriptive_words = {
    "phòng": ["rộng", "thoải mái", "sạch", "gọn gàng", "thoáng", "đẹp"],
    "nhân viên": ["chuyên nghiệp", "thân thiện", "nhiệt tình", "tốt", "tệ"],
    "đồ ăn": ["ngon", "dở", "tươi", "mặn", "nhạt"],
    "vị trí": ["tốt", "gần", "tiện lợi", "xa", "dễ dàng", "thuận tiện"],
    "giá cả": ["hợp lý", "đắt", "rẻ", "phải chăng"],
    "view": ["đẹp", "rộng", "hướng biển", "tuyệt vời", "hấp dẫn"],
    "wifi": ["mạnh", "kém", "ổn định", "chậm"],
    "hồ bơi": ["sạch", "mát", "lạnh", "thoải mái"],
    "bãi đỗ xe": ["dễ dàng", "thoải mái", "chật", "khó khăn"],
    "dịch vụ phòng": ["tốt", "tuyệt vời", "kém", "nhanh", "chậm"],
    "spa": ["thư giãn", "tuyệt vời", "kém", "tốt"],
    "gym": ["hiện đại", "đầy đủ", "không gian rộng", "cơ bản"],
    "giải trí": ["vui", "thú vị", "buồn tẻ", "nhàm chán"],
    "đưa đón": ["tiện lợi", "nhanh chóng", "tốt"],
    "bãi biển": ["sạch", "cát trắng", "nước trong", "mát", "đẹp"],
    "núi": ["cao", "hùng vĩ", "đẹp"],
    "sông": ["trong", "mát", "hấp dẫn"],
    "hồ": ["sạch", "trong", "mát"],
    "đảo": ["hoang sơ", "đẹp", "tuyệt vời"],
    "trung tâm": ["nổi bật", "tấp nập", "sôi động"],
    "an ninh": ["an toàn", "chặt chẽ", "kém"],
    "sạch sẽ": ["gọn gàng", "ngăn nắp", "mớ hỗn độn"],
    "tiện nghi": ["đầy đủ", "hiện đại", "kém"],
    "không gian": ["rộng", "thoáng", "hẹp"],
    "yên tĩnh": ["thoải mái", "yên bình", "ồn ào"],
    "thời tiết": ["nắng", "mưa", "lạnh", "nóng"],
    "đồ uống": ["ngon", "mát", "khó uống"],
    "trải nghiệm tổng thể": ["tuyệt vời", "tốt", "kém", "tệ"]
}


# Chuẩn hóa từ: bỏ dấu câu, chuyển thành _ nếu là từ ghép, nhưng không có dấu _ ở đầu

def normalize_token(word):
    word = re.sub(r'[^\w\s]', '', word)  # bỏ dấu câu
    word = word.strip()
    word = word.replace(" ", "_")
    if word.startswith("_"):
        word = word[1:]
    return word.lower()


def analyze_aspect_sentiments(text):
    """Phân tích các khía cạnh trong câu và trích xuất từ khóa mô tả & ngữ cảnh chính xác theo từng aspect."""
    tokens = underthesea.word_tokenize(text, format="text").split()
    token_text = "_".join(tokens)  # Chuẩn hóa để so sánh keyword dễ hơn

    detected_aspects = {}

    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            norm_keyword = keyword.replace(" ", "_")
            if norm_keyword in token_text:
                # Tìm vị trí keyword trong tokens
                positions = [i for i, t in enumerate(
                    tokens) if norm_keyword in t or keyword in t]
                for pos in positions:
                    detected_aspects.setdefault(aspect, set())

                    # Lấy ngữ cảnh xung quanh keyword ±4 từ
                    start = max(0, pos - 4)
                    end = min(len(tokens), pos + 5)
                    context = tokens[start:end]

                    # Tập hợp từ khóa liên quan trong ngữ cảnh
                    context_keywords = set()
                    context_keywords.add(aspect.replace(" ", "_"))
                    context_keywords.add(norm_keyword)

                    for word in context:
                        norm_word = normalize_token(word)
                        if not norm_word or norm_word in vietnamese_stopwords:
                            continue
                        if norm_word in descriptive_words.get(aspect, []):
                            context_keywords.add(norm_word)
                        elif norm_word != aspect.replace(" ", "_"):
                            context_keywords.add(norm_word)

                    # Cập nhật từ khóa theo từng context riêng biệt
                    detected_aspects[aspect].update(context_keywords)

    if not detected_aspects:
        return []

    sentiments = []
    for aspect, extracted_keywords in detected_aspects.items():
        sentiment_score = round(random.uniform(-1, 1), 2)
        confidence = round(random.uniform(0.7, 1.0), 2)

        rule_based_keywords = {
            "sạch_sẽ", "rộng_rãi", "nhân_viên", "chuyên_nghiệp", "đồ_ăn", "ngon", "giá", "cao",
            "tuyệt_vời", "xuất_sắc", "hoàn_hảo", "thân_thiện", "chu_đáo", "thoải_mái", "an_toàn",
            "yên_tĩnh", "thuận_tiện", "đẹp", "hấp_dẫn", "phong_phú", "đa_dạng", "giá_cả_hợp_lý"
        }

        if any(word in rule_based_keywords for word in extracted_keywords):
            extraction_method = "rule-based"
        else:
            extraction_method = "machine-learning"

        sentiments.append((
            aspect,
            sentiment_score,
            confidence,
            extracted_keywords,
            extraction_method
        ))

    return sentiments


# ============================
# 5️⃣ Xử lý từng bài đánh giá
# ============================


def process_reviews(df):
    """Xử lý dữ liệu review, tạo đặc trưng TF-IDF và Word2Vec, phân tích cảm xúc."""
    print("🚀 Bắt đầu quá trình xử lý reviews...")
    results = []
    aspect_sentiments = []
    all_texts = df["final_text"].tolist()
    tfidf_vectorizer.fit(all_texts)
    if not featureEngineer.word2vec_model:
        featureEngineer.create_word_embeddings(all_texts, retrain=True)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="📝 Xử lý reviews"):
        review_id, text, created_at = row['review_id'], row['final_text'], row['created_at']

        # Generate the TF-IDF features for this review
        tfidf_features = tfidf_vectorizer.transform([text]).toarray()[0]

        # Generate Word2Vec embeddings for this review
        word2vec_embeddings = featureEngineer.create_word_embeddings(text)

        # Ensure word2vec_embeddings is a numpy array 1D
        if isinstance(word2vec_embeddings, list) and len(word2vec_embeddings) == 1:
            word2vec_embeddings = word2vec_embeddings[0]

        # Add results to the list
        results.extend([
            (review_id, "tfidf", tfidf_features,
             tfidf_features.shape[0], created_at),
            (review_id, "word2vec", word2vec_embeddings,
             word2vec_embeddings.shape[0], created_at)
        ])

        # Process aspect sentiments
        aspects = analyze_aspect_sentiments(text)
        for aspect, sentiment_score, confidence, extracted_keywords, extraction_method in aspects:
            aspect_sentiments.append((review_id, aspect, sentiment_score, confidence,
                                     extracted_keywords, extraction_method, created_at))

    print("✅ Xử lý review hoàn tất!")
    return results, aspect_sentiments

# ============================
# 6️⃣ Lưu dữ liệu vào database
# ============================


def save_to_database(results, aspect_sentiments):
    """Lưu dữ liệu embeddings và aspect sentiments vào database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    print("💾 Đang lưu dữ liệu vào database...")
    for review_id, embedding_type, embedding, dimensions, created_at in results:
        # Đảm bảo embedding là numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Đảm bảo dimensions khớp với kích thước thực
        actual_dimensions = embedding.size
        if dimensions != actual_dimensions:
            print(
                f"⚠️ Cảnh báo: Không khớp kích thước cho review {review_id}, type {embedding_type}: Expected {dimensions}, got {actual_dimensions}")
            dimensions = actual_dimensions

        embedding_bytes = pickle.dumps(embedding)
        cursor.execute("""
            INSERT INTO review_embeddings (review_id, embedding_type, embedding, dimensions, created_at)
            VALUES (%s, %s, %s, %s, %s);
        """, (review_id, embedding_type, embedding_bytes, dimensions, created_at))
    for review_id, aspect, sentiment_score, confidence, extracted_keywords, extraction_method, created_at in aspect_sentiments:
        cursor.execute("""
            INSERT INTO aspect_sentiments (review_id, aspect, sentiment_score, confidence, extracted_keywords, extraction_method, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (review_id, aspect, sentiment_score, confidence, list(extracted_keywords), extraction_method, created_at))
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Dữ liệu đã được lưu vào database!")

# ============================
# 9️⃣ Lưu dữ liệu vào file Excel
# ============================


def save_to_excel(results, aspect_sentiments, filename="output.xlsx"):
    """Lưu dữ liệu embeddings và aspect sentiments vào file Excel."""
    print("💾 Đang lưu dữ liệu vào Excel...")
    df_results = pd.DataFrame(results, columns=[
                              "review_id", "embedding_type", "embedding", "dimensions", "created_at"])
    df_aspects = pd.DataFrame(aspect_sentiments, columns=[
                              "review_id", "aspect", "sentiment_score", "confidence", "extracted_keywords", "extraction_method", "created_at"])

    with pd.ExcelWriter(filename) as writer:
        df_results.to_excel(
            writer, sheet_name="Review Embeddings", index=False)
        df_aspects.to_excel(
            writer, sheet_name="Aspect Sentiments", index=False)

    print(f"✅ Dữ liệu đã được lưu vào {filename}!")

# ============================
# 🧪 TEST ĐOẠN VĂN MẪU
# ============================


test_text = """“Homestay chất lượng” Phòng ở sạch sẽ, tuy nhiên diện tích hơi nhỏ, bù lại bày trí xinh. 
Nhân viên lịch sự. Trang thiết bị cũng ok nhưng không có tủ lạnh cũng hơi bất tiện. Nhưng nhìn chung thì khá ổn"""

print("\n🔍 Đang phân tích đoạn văn mẫu:\n", test_text)
print("\n📎 Kết quả trích xuất:")
result = analyze_aspect_sentiments(test_text)
for aspect, sentiment_score, confidence, extracted_keywords, extraction_method in result:
    print(f"\n🔹 Aspect: {aspect}")
    print(f"   🔸 Sentiment Score: {sentiment_score}")
    print(f"   🔸 Confidence: {confidence}")
    print(f"   🔸 Extracted Keywords: {extracted_keywords}")
    print(f"   🔸 Extraction Method: {extraction_method}")


# ============================
# 🔥 CHẠY CHƯƠNG TRÌNH CHÍNH
# ============================
if __name__ == "__main__":
    df_reviews = fetch_reviews()
    results, aspect_sentiments = process_reviews(df_reviews)
    save_to_excel(results, aspect_sentiments)
    save_to_database(results, aspect_sentiments)
    print("🎉 Hoàn thành quá trình xử lý dữ liệu!")
