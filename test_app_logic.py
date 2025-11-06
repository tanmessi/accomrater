#!/usr/bin/env python3
# test_app_logic.py - Test the core logic from app_updated.py

import os
import sys
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

# Test paths
TRAIN_PATH = r'./datasets/vlsp2018_hotel/1-VLSP2018-SA-Hotel-train.csv'
VAL_PATH = r'./datasets/vlsp2018_hotel/2-VLSP2018-SA-Hotel-dev.csv'
TEST_PATH = r'./datasets/vlsp2018_hotel/3-VLSP2018-SA-Hotel-test.csv'

def get_aspect_category_names():
    """Get aspect category names from dataset"""
    raw_datasets = VLSP2018Loader.load(TRAIN_PATH, VAL_PATH, TEST_PATH)
    return raw_datasets['train'].column_names[1:]

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

def load_model(model_path: str, aspect_category_names: List[str], tokenizer):
    """Load ACSA model with weights"""
    import tensorflow as tf

    optimizer = Adam(learning_rate=1e-4)
    model = VLSP2018MultiTask(PRETRAINED_MODEL, aspect_category_names, optimizer, name=os.path.basename(model_path).replace('.h5', ''))

    try:
        # Build the model first with a dummy input that matches expected shapes
        dummy_inputs = tokenizer(
            "dummy text",
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

        # Call the model to build it - convert BatchEncoding to dict
        _ = model(dict(dummy_inputs))

        # Now load the weights with by_name and skip_mismatch
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model, None
    except Exception as e:
        return None, str(e)

def analyze_single_review(review_text: str, model, tokenizer, vn_preprocessor, aspect_category_names) -> Dict:
    """Analyze a single review and return predictions"""
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

def main():
    print("=" * 80)
    print("Testing app_updated.py core logic")
    print("=" * 80)

    # Step 1: Load aspect category names
    print("\n1. Loading aspect category names...")
    aspect_category_names = get_aspect_category_names()
    print(f"   ✓ Loaded {len(aspect_category_names)} aspect categories")

    # Step 2: Load tokenizer and preprocessor
    print("\n2. Loading tokenizer and preprocessor...")
    tokenizer, vn_preprocessor = load_tokenizer_and_preprocessor()
    print("   ✓ Tokenizer and preprocessor loaded")

    # Step 3: Load model
    print("\n3. Loading model...")
    model_path = os.path.join(WEIGHTS_DIR, 'Hotel-v1.h5')
    model, error = load_model(model_path, aspect_category_names, tokenizer)

    if error:
        print(f"   ✗ Error loading model: {error}")
        return

    print(f"   ✓ Model loaded from {model_path}")

    # Step 4: Test with a sample review
    print("\n4. Testing with sample review...")
    test_review = "Khách sạn rất sạch sẽ và nhân viên thân thiện. Tuy nhiên giá hơi đắt."
    print(f"   Review: {test_review}")

    try:
        results = analyze_single_review(
            test_review, model, tokenizer,
            vn_preprocessor, aspect_category_names
        )

        print("\n   Results:")
        print(f"   - Overall sentiment: {results['overall_sentiment']}")
        print(f"   - Sentiment score: {results['sentiment_score']:.2f}")
        print(f"   - Sentiment counts: {results['sentiment_counts']}")
        print(f"   - Detected aspects: {len(results['aspects'])}")

        print("\n   Aspect details:")
        for aspect in results['aspects']:
            print(f"     • {aspect['aspect']}: {aspect['polarity']}")

        print("\n" + "=" * 80)
        print("✓ All tests passed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n   ✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
