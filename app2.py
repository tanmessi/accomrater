# app_updated.py - Updated Streamlit App with new ACSA model logic

import streamlit as st
import sys
import os
import pandas as pd
from typing import List, Dict
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam

# Project imports
from processors.vlsp2018_processor import VLSP2018Loader, PolarityMapping
from processors.vietnamese_processor import VietnameseTextPreprocessor
from transformers import AutoTokenizer
from acsa_model import VLSP2018MultiTask

# Constants
PRETRAINED_MODEL = 'vinai/phobert-base'
MAX_LENGTH = 256
WEIGHTS_DIR = './weights'

# Streamlit page config
st.set_page_config(
    page_title="Phân tích cảm xúc review khách sạn",
    page_icon="🏨",
    layout="wide"
)

# Cache resources
@st.cache_resource
def load_tokenizer_and_preprocessor():
    """Load tokenizer and Vietnamese preprocessor"""
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    vn_preprocessor = VietnameseTextPreprocessor(
        vncorenlp_dir='processors/VnCoreNLP',
        extra_teencodes={
            'khách sạn': ['ks'], 'nhà hàng': ['nhahang'], 'nhân viên': ['nv'],
            'cửa hàng': ['store', 'sop', 'shopE', 'shop'],
            'sản phẩm': ['sp', 'product'], 'hàng': ['hàg'],
            'giao hàng': ['ship', 'delivery', 'síp'], 'đặt hàng': ['order'],
            'chuẩn chính hãng': ['authentic', 'aut', 'auth'], 'hạn sử dụng': ['date', 'hsd'],
            'điện thoại': ['dt'], 'facebook': ['fb', 'face'],
            'nhắn tin': ['nt', 'ib'], 'trả lời': ['tl', 'trl', 'rep'],
            'feedback': ['fback', 'fedback'], 'sử dụng': ['sd'], 'xài': ['sài'],
        },
        max_correction_length=MAX_LENGTH
    )
    return tokenizer, vn_preprocessor

@st.cache_resource
def get_aspect_category_names():
    """Get aspect category names from dataset"""
    TRAIN_PATH = r'./datasets/vlsp2018_hotel/1-VLSP2018-SA-Hotel-train.csv'
    VAL_PATH = r'./datasets/vlsp2018_hotel/2-VLSP2018-SA-Hotel-dev.csv'
    TEST_PATH = r'./datasets/vlsp2018_hotel/3-VLSP2018-SA-Hotel-test.csv'

    raw_datasets = VLSP2018Loader.load(TRAIN_PATH, VAL_PATH, TEST_PATH)
    return raw_datasets['train'].column_names[1:]

@st.cache_resource
def load_model(model_path: str, aspect_category_names: List[str], _tokenizer):
    """Load ACSA model with weights - following test_app_logic.py structure exactly"""
    import tensorflow as tf

    # Extract model name from path
    model_name = os.path.basename(model_path).replace('.h5', '')

    # Create optimizer (same as test_app_logic.py line 54)
    optimizer = Adam(learning_rate=1e-4)

    # Instantiate model (same as test_app_logic.py line 55)
    # Do NOT use multi_branch parameter - use default
    model = VLSP2018MultiTask(PRETRAINED_MODEL, aspect_category_names, optimizer, name=model_name)

    try:
        # Build the model first with a dummy input (same as test_app_logic.py line 58-68)
        dummy_inputs = _tokenizer(
            "dummy text",
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

        # Call the model to build it - convert BatchEncoding to dict
        _ = model(dict(dummy_inputs))

        # Load the weights with by_name and skip_mismatch (CRITICAL - line 71)
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model, None

    except Exception as e:
        return None, str(e)

def get_available_models() -> List[str]:
    """Get list of available model weights"""
    if not os.path.exists(WEIGHTS_DIR):
        return []

    models = []
    for item in os.listdir(WEIGHTS_DIR):
        if item.endswith('.h5') or os.path.isdir(os.path.join(WEIGHTS_DIR, item)):
            models.append(item)

    return sorted(models)

def analyze_single_review(review_text: str, model, tokenizer, vn_preprocessor, aspect_category_names) -> Dict:
    """
    Analyze a single review and return predictions

    Returns:
        Dict with keys: 'aspects', 'overall_sentiment', 'sentiment_score'
    """
    # Preprocess and tokenize
    processed_input = VLSP2018Loader.preprocess_and_tokenize(
        review_text, vn_preprocessor, tokenizer,
        batch_size=1, max_length=MAX_LENGTH
    )

    # Create TensorFlow dataset
    tf_inputs = Dataset.from_tensor_slices({
        x: [[processed_input[x][0]]] for x in tokenizer.model_input_names
    })

    # Predict
    predictions = model.acsa_predict(tf_inputs)

    # Parse results
    aspects = []
    polarities_map = {0: 'neutral', 1: 'positive', 2: 'negative', 3: 'none'}

    for aspect_category, polarity_idx in zip(aspect_category_names, predictions[0]):
        polarity = PolarityMapping.INDEX_TO_POLARITY.get(polarity_idx, 'none')
        if polarity and polarity != 'none':
            aspects.append({
                'aspect': aspect_category,
                'polarity': polarity,
                'polarity_idx': polarity_idx
            })

    # Calculate overall sentiment score
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for aspect in aspects:
        if aspect['polarity'] in sentiment_counts:
            sentiment_counts[aspect['polarity']] += 1

    total = sum(sentiment_counts.values())
    if total > 0:
        sentiment_score = (sentiment_counts['positive'] - sentiment_counts['negative']) / total

        if sentiment_score > 0.3:
            overall_sentiment = 'Tích cực'
        elif sentiment_score < -0.3:
            overall_sentiment = 'Tiêu cực'
        else:
            overall_sentiment = 'Trung bình'
    else:
        sentiment_score = 0
        overall_sentiment = 'Không xác định'

    return {
        'aspects': aspects,
        'overall_sentiment': overall_sentiment,
        'sentiment_score': sentiment_score,
        'sentiment_counts': sentiment_counts
    }

def display_single_analysis_results(results: Dict, use_expander: bool = True):
    """Display results for single review analysis

    Args:
        results: Analysis results dictionary
        use_expander: If False, display aspects without nested expanders (for use inside other expanders)
    """
    # Overall sentiment
    st.markdown("### 📊 Kết quả phân tích tổng thể")

    col1, col2, col3 = st.columns(3)

    with col1:
        sentiment_color = {
            'Tích cực': '#00c853',
            'Tiêu cực': '#d50000',
            'Trung bình': '#ffc107',
            'Không xác định': '#9e9e9e'
        }.get(results['overall_sentiment'], '#9e9e9e')

        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {sentiment_color}20; border: 2px solid {sentiment_color};'>
            <h3 style='color: {sentiment_color}; margin: 0;'>{results['overall_sentiment']}</h3>
            <p style='margin: 5px 0 0 0; color: #666;'>Cảm xúc tổng thể</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Điểm cảm xúc", f"{results['sentiment_score']:.2f}")

    with col3:
        st.markdown("**Phân bố:**")
        st.write(f"✅ Tích cực: {results['sentiment_counts']['positive']}")
        st.write(f"❌ Tiêu cực: {results['sentiment_counts']['negative']}")
        st.write(f"➖ Trung bình: {results['sentiment_counts']['neutral']}")

    # Aspects details
    st.markdown("### 🔍 Chi tiết theo từng khía cạnh")

    if results['aspects']:
        # Group by category
        categories = {}
        for aspect in results['aspects']:
            category = aspect['aspect'].split('#')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(aspect)

        # Display by category
        for category, aspects_list in categories.items():
            if use_expander:
                # Use expander when displaying standalone (single review mode)
                with st.expander(f"**{category}** ({len(aspects_list)} khía cạnh)", expanded=True):
                    for aspect in aspects_list:
                        polarity_emoji = {
                            'positive': '✅',
                            'negative': '❌',
                            'neutral': '➖'
                        }.get(aspect['polarity'], '❓')

                        polarity_color = {
                            'positive': '#00c853',
                            'negative': '#d50000',
                            'neutral': '#ffc107'
                        }.get(aspect['polarity'], '#9e9e9e')

                        st.markdown(f"""
                        <div style='padding: 10px; margin: 5px 0; border-left: 3px solid {polarity_color}; background-color: {polarity_color}15;'>
                            {polarity_emoji} <strong>{aspect['aspect'].split('#')[1]}</strong>: <span style='color: {polarity_color};'>{aspect['polarity']}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Display without expander (when already inside an expander)
                st.markdown(f"**{category}** ({len(aspects_list)} khía cạnh)")
                for aspect in aspects_list:
                    polarity_emoji = {
                        'positive': '✅',
                        'negative': '❌',
                        'neutral': '➖'
                    }.get(aspect['polarity'], '❓')

                    polarity_color = {
                        'positive': '#00c853',
                        'negative': '#d50000',
                        'neutral': '#ffc107'
                    }.get(aspect['polarity'], '#9e9e9e')

                    st.markdown(f"""
                    <div style='padding: 10px; margin: 5px 0; border-left: 3px solid {polarity_color}; background-color: {polarity_color}15;'>
                        {polarity_emoji} <strong>{aspect['aspect'].split('#')[1]}</strong>: <span style='color: {polarity_color};'>{aspect['polarity']}</span>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Không phát hiện khía cạnh nào trong review")

    # Conclusion
    st.markdown("### 📝 Kết luận")
    if results['overall_sentiment'] == 'Tích cực':
        st.success(f"Review này thể hiện cảm xúc tích cực với {results['sentiment_counts']['positive']} khía cạnh được đánh giá cao.")
    elif results['overall_sentiment'] == 'Tiêu cực':
        st.error(f"Review này thể hiện cảm xúc tiêu cực với {results['sentiment_counts']['negative']} khía cạnh cần cải thiện.")
    else:
        st.warning("Review này có cảm xúc trung bình hoặc không rõ ràng.")

def main():
    st.title("🏨 Phân tích cảm xúc review khách sạn với ACSA")

    # Sidebar
    st.sidebar.header("⚙️ Cài đặt")

    # Model selection
    available_models = get_available_models()
    if not available_models:
        st.sidebar.error("Không tìm thấy model weights trong folder ./weights/")
        st.error("Vui lòng kiểm tra folder weights/")
        return

    selected_model = st.sidebar.selectbox(
        "Chọn mô hình",
        available_models,
        help="Chọn model đã được train để phân tích"
    )

    # Build correct model path
    if selected_model.endswith('.h5'):
        model_path = os.path.join(WEIGHTS_DIR, selected_model)
    else:
        # For folder-based models, look for model files inside
        model_path = os.path.join(WEIGHTS_DIR, selected_model, selected_model)

    # Load resources
    with st.spinner("Đang tải model và preprocessor..."):
        tokenizer, vn_preprocessor = load_tokenizer_and_preprocessor()
        aspect_category_names = get_aspect_category_names()
        model, error = load_model(model_path, aspect_category_names, tokenizer)

    if error:
        st.sidebar.error(f"Lỗi tải model: {error}")
        st.error("Không thể tải model. Vui lòng kiểm tra file weights.")
        return

    st.sidebar.success(f"✅ Đã tải model: {selected_model}")
    st.sidebar.info(f"📋 Số lượng khía cạnh: {len(aspect_category_names)}")

    # Main content
    st.markdown("---")

    # Mode selection
    analysis_mode = st.radio(
        "Chế độ phân tích",
        ["Nhập một review", "Nhập nhiều reviews"],
        horizontal=True
    )

    if analysis_mode == "Nhập một review":
        st.markdown("### 📝 Nhập nội dung review")

        review_text = st.text_area(
            "Review",
            height=150,
            placeholder="Nhập review của bạn tại đây...",
            help="Nhập nội dung đánh giá về khách sạn"
        )

        if st.button("🔍 Phân tích review", type="primary"):
            if not review_text.strip():
                st.warning("Vui lòng nhập nội dung review")
            else:
                with st.spinner("Đang phân tích review..."):
                    try:
                        results = analyze_single_review(
                            review_text, model, tokenizer,
                            vn_preprocessor, aspect_category_names
                        )

                        st.markdown("---")
                        display_single_analysis_results(results)

                    except Exception as e:
                        st.error(f"Lỗi khi phân tích: {str(e)}")
                        st.exception(e)

    else:  # Nhập nhiều reviews
        st.markdown("### 📋 Nhập nhiều reviews")

        # Option 1: Text area with multiple lines
        reviews_text = st.text_area(
            "Nhập các reviews (mỗi review một dòng)",
            height=200,
            placeholder="Review 1...\nReview 2...\nReview 3...",
            help="Mỗi dòng là một review riêng biệt"
        )

        # Option 2: File upload
        uploaded_file = st.file_uploader(
            "Hoặc upload file CSV/TXT",
            type=['csv', 'txt'],
            help="File CSV cần có cột 'review' hoặc file TXT mỗi dòng là một review"
        )

        if st.button("🔍 Phân tích tất cả reviews", type="primary"):
            reviews_list = []

            # Get reviews from text area
            if reviews_text.strip():
                reviews_list = [r.strip() for r in reviews_text.split('\n') if r.strip()]

            # Get reviews from file
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'review' in df.columns:
                        reviews_list.extend(df['review'].dropna().tolist())
                    else:
                        st.error("File CSV cần có cột 'review'")
                else:  # txt file
                    content = uploaded_file.read().decode('utf-8')
                    reviews_list.extend([r.strip() for r in content.split('\n') if r.strip()])

            if not reviews_list:
                st.warning("Không có review nào để phân tích")
            else:
                st.info(f"Tìm thấy {len(reviews_list)} reviews")

                # Analyze all reviews
                all_results = []
                progress_bar = st.progress(0)

                for idx, review in enumerate(reviews_list):
                    try:
                        result = analyze_single_review(
                            review, model, tokenizer,
                            vn_preprocessor, aspect_category_names
                        )
                        result['review_text'] = review
                        result['review_id'] = idx + 1
                        all_results.append(result)
                    except Exception as e:
                        st.warning(f"Lỗi khi phân tích review {idx+1}: {str(e)}")

                    progress_bar.progress((idx + 1) / len(reviews_list))

                progress_bar.empty()

                # Display summary
                st.markdown("---")
                st.markdown("### 📊 Tổng quan kết quả")

                total_positive = sum(1 for r in all_results if r['overall_sentiment'] == 'Tích cực')
                total_negative = sum(1 for r in all_results if r['overall_sentiment'] == 'Tiêu cực')
                total_neutral = sum(1 for r in all_results if r['overall_sentiment'] == 'Trung bình')

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Tổng reviews", len(all_results))
                col2.metric("✅ Tích cực", total_positive)
                col3.metric("❌ Tiêu cực", total_negative)
                col4.metric("➖ Trung bình", total_neutral)

                # Show individual results
                st.markdown("### 📋 Chi tiết từng review")

                for result in all_results:
                    with st.expander(f"Review #{result['review_id']}: {result['overall_sentiment']}", expanded=False):
                        st.write(f"**Nội dung:** {result['review_text'][:200]}...")
                        # Use use_expander=False to avoid nested expanders
                        display_single_analysis_results(result, use_expander=False)

if __name__ == "__main__":
    main()
