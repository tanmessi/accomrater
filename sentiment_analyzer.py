import os
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to get more information
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class SentimentAnalyzer:
    def __init__(self, 
                 emotion_words_path='emotion_words.csv', 
                 aspect_keywords_path='aspect_keywords.yaml'):
        """
        Initialize the SentimentAnalyzer with emotion words and aspect keywords.
        
        Args:
            emotion_words_path (str): Path to CSV file with emotion words
            aspect_keywords_path (str): Path to YAML file with aspect keywords
        """
        # Validate input files
        if not os.path.exists(emotion_words_path):
            raise FileNotFoundError(f"Emotion words file not found: {emotion_words_path}")
        if not os.path.exists(aspect_keywords_path):
            raise FileNotFoundError(f"Aspect keywords file not found: {aspect_keywords_path}")
        
        # Load emotion words
        self.emotion_words = self.load_emotion_words(emotion_words_path)
        
        # Load aspect keywords
        with open(aspect_keywords_path, 'r', encoding='utf-8') as f:
            self.aspect_keywords = yaml.safe_load(f)
        
        logging.info(f"Loaded {len(self.emotion_words)} emotion words")
        logging.info(f"Loaded aspect keywords for {len(self.aspect_keywords)} aspects")

    def load_emotion_words(self, file_path):
        """
        Load emotion words from CSV file.
        
        Args:
            file_path (str): Path to CSV file with emotion words
        
        Returns:
            dict: Dictionary of emotion words with their sentiment scores
        """
        emotion_words = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    emotion_words[row['word']] = float(row['score'])
        except Exception as e:
            logging.error(f"Error loading emotion words: {e}")
            raise
        return emotion_words

    def print_yaml_structure(self, input_yaml_path):
        """
        Print detailed information about the YAML file structure for debugging.
        
        Args:
            input_yaml_path (str): Path to input YAML file
        """
        try:
            with open(input_yaml_path, 'r', encoding='utf-8') as f:
                reviews = yaml.safe_load(f)
            
            logging.info(f"Total reviews loaded: {len(reviews)}")
            logging.info("Sample review structure:")
            if reviews:
                sample_review = reviews[0]
                logging.info(f"Sample review type: {type(sample_review)}")
                logging.info(f"Sample review keys: {sample_review.keys() if isinstance(sample_review, dict) else 'N/A'}")
        except Exception as e:
            logging.error(f"Error examining YAML file: {e}")

    def normalize_review_dict(self, review):
        """
        Normalize review dictionary to ensure consistent structure.
        
        Args:
            review (dict or str): Review data
        
        Returns:
            dict: Normalized review dictionary or None
        """
        # If review is a string, try to convert to dictionary
        if isinstance(review, str):
            try:
                review = yaml.safe_load(review)
            except Exception as e:
                logging.error(f"Could not parse review string: {e}")
                return None
        
        # Ensure review is a dictionary
        if not isinstance(review, dict):
            logging.error(f"Invalid review format: {type(review)}")
            return None
        
        # Log detailed review structure for debugging
        logging.debug(f"Review keys: {review.keys()}")
        
        # Check and set default values for required fields
        required_fields = [
            'review_id', 'aspect', 'final_text', 
            'extracted_keywords', 'hotel_id'
        ]
        
        for field in required_fields:
            if field not in review:
                logging.warning(f"Missing field '{field}' in review")
                review[field] = None
        
        return review

    def calculate_sentiment_score(self, review):
        """
        Calculate sentiment score for a single review.
        
        Args:
            review (dict): Review dictionary
        
        Returns:
            float or None: Calculated sentiment score
        """
        # Normalize review dictionary
        review = self.normalize_review_dict(review)
        if review is None:
            return None
        
        # Extract necessary fields
        final_text = str(review.get('final_text', '')).lower().replace('_', ' ')
        aspect = review.get('aspect', '')
        keywords = review.get('extracted_keywords', [])
        review_id = review.get('review_id', 'Unknown')
        
        # Extensive logging for debugging
        logging.debug(f"Processing review: {review_id}")
        logging.debug(f"Aspect: {aspect}")
        logging.debug(f"Keywords: {keywords}")
        
        # If no keywords or aspect, return None
        if not keywords or not aspect:
            logging.warning(f"No keywords or aspect for review: {review_id}")
            return None
        
        # Validate aspect exists in keywords
        if aspect not in self.aspect_keywords:
            logging.warning(f"Unknown aspect '{aspect}' for review: {review_id}")
            # Log available aspects for reference
            logging.warning(f"Available aspects: {list(self.aspect_keywords.keys())}")
            return None
        
        # Get positive and negative keywords for this aspect
        aspect_positive = self.aspect_keywords[aspect].get('positive', [])
        aspect_negative = self.aspect_keywords[aspect].get('negative', [])
        
        # Count keyword matches
        pos_count = 0
        neg_count = 0
        for keyword_obj in keywords:
            # Ensure keyword_obj is a dictionary
            if not isinstance(keyword_obj, dict):
                logging.warning(f"Invalid keyword object for review {review_id}")
                continue
            
            keyword = str(keyword_obj.get('keyword', '')).replace('_', ' ')
            if keyword in aspect_positive:
                pos_count += 1
            elif keyword in aspect_negative:
                neg_count += 1
        
        # Calculate score from keywords
        keyword_score = (pos_count - neg_count) / (pos_count + neg_count + 1) if (pos_count + neg_count) > 0 else 0
        
        # Calculate emotion score
        emotion_score = 0
        for word, score in self.emotion_words.items():
            if word in final_text:
                emotion_score += score
        
        # Combine scores
        sentiment_score = keyword_score + emotion_score
        
        # Normalize to [0, 1] range
        normalized_score = (sentiment_score + 1) / 2
        
        # Ensure score is within [0, 1]
        final_score = max(0, min(1, normalized_score))
        
        logging.debug(f"Review {review_id}: keyword_score={keyword_score}, emotion_score={emotion_score}, final_score={final_score}")
        
        return final_score

    def process_reviews(self, input_yaml_path):
        """
        Process reviews and calculate sentiment scores.
        
        Args:
            input_yaml_path (str): Path to input YAML file with reviews
        
        Returns:
            list: Processed reviews with sentiment scores
        """
        # Validate input file
        if not os.path.exists(input_yaml_path):
            raise FileNotFoundError(f"Input YAML file not found: {input_yaml_path}")
        
        # Print detailed YAML structure for debugging
        self.print_yaml_structure(input_yaml_path)
        
        # Load reviews from YAML
        try:
            with open(input_yaml_path, 'r', encoding='utf-8') as f:
                reviews = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading reviews from YAML: {e}")
            raise
        
        # Validate reviews is a list
        if not isinstance(reviews, list):
            logging.error(f"Reviews is not a list, but {type(reviews)}")
            reviews = [reviews] if reviews is not None else []
        
        # Calculate sentiment scores
        processed_reviews = []
        total_reviews = len(reviews)
        processed_count = 0
        
        for review in reviews:
            try:
                # Calculate sentiment score
                sentiment_score = self.calculate_sentiment_score(review)
                
                # Update review with new sentiment score
                if sentiment_score is not None:
                    if isinstance(review, dict):
                        review['calculated_sentiment_score'] = round(sentiment_score, 2)
                    else:
                        review = {'original_review': review, 'calculated_sentiment_score': round(sentiment_score, 2)}
                    
                    processed_reviews.append(review)
                    processed_count += 1
                else:
                    logging.warning(f"Skipping review due to no sentiment score")
            except Exception as e:
                logging.error(f"Error processing review: {e}")
                logging.error(f"Review details: {review}")
        
        logging.info(f"Processed {processed_count}/{total_reviews} reviews")
        
        # Save processed reviews
        try:
            with open('processed_reviews_with_sentiment.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(processed_reviews, f, allow_unicode=True)
        except Exception as e:
            logging.error(f"Error saving processed reviews: {e}")
        
        return processed_reviews

    def generate_visualizations(self, processed_reviews):
        """
        Generate visualizations for sentiment scores and other metrics.
        
        Args:
            processed_reviews (list): List of processed reviews
        """
        # Extract calculated sentiment scores
        sentiment_scores = []
        aspects = []
        hotel_ids = []
        
        for review in processed_reviews:
            # Handle different review formats
            if isinstance(review, dict):
                score = review.get('calculated_sentiment_score')
                aspect = review.get('aspect')
                hotel_id = review.get('hotel_id')
                
                if score is not None:
                    sentiment_scores.append(score)
                    aspects.append(aspect)
                    hotel_ids.append(hotel_id)
        
        # Check if we have any processed reviews
        if not sentiment_scores:
            logging.error("No reviews processed. Cannot generate visualizations.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'sentiment_score': sentiment_scores,
            'aspect': aspects,
            'hotel_id': hotel_ids
        })
        
        # Create a figure with subplots
        plt.figure(figsize=(15, 10))
        
        # 1. Sentiment Score Distribution
        plt.subplot(2, 2, 1)
        df['sentiment_score'].hist(bins=20, edgecolor='black')
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        
        # 2. Sentiment Score by Hotel (if hotels exist)
        plt.subplot(2, 2, 2)
        hotel_sentiment = df.groupby('hotel_id')['sentiment_score'].mean()
        
        if not hotel_sentiment.empty:
            hotel_sentiment.plot(kind='bar', edgecolor='black')
            plt.title('Average Sentiment Score by Hotel')
            plt.xlabel('Hotel ID')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(rotation=45, ha='right')
        else:
            plt.text(0.5, 0.5, 'No Hotel Data', 
                     horizontalalignment='center', 
                     verticalalignment='center')
            plt.title('No Hotel Sentiment Data')
        
        # 3. Aspect Distribution
        plt.subplot(2, 2, 3)
        df['aspect'].value_counts().plot(kind='bar', edgecolor='black')
        plt.title('Distribution of Aspects')
        plt.xlabel('Aspect')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # 4. Sentiment Score by Aspect
        plt.subplot(2, 2, 4)
        aspect_sentiment = df.groupby('aspect')['sentiment_score'].mean()
        if not aspect_sentiment.empty:
            aspect_sentiment.plot(kind='bar', edgecolor='black')
            plt.title('Average Sentiment Score by Aspect')
            plt.xlabel('Aspect')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(rotation=45, ha='right')
        else:
            plt.text(0.5, 0.5, 'No Aspect Sentiment Data', 
                     horizontalalignment='center', 
                     verticalalignment='center')
            plt.title('No Aspect Sentiment Data')
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis_visualizations.png')
        logging.info("Visualizations saved to sentiment_analysis_visualizations.png")
        
        # Save numerical data to CSVs
        # Sentiment Scores
        df.to_csv('sentiment_scores.csv', index=False)
        
        # Aspects Distribution
        aspect_dist = df['aspect'].value_counts().reset_index()
        aspect_dist.columns = ['Aspect', 'Count']
        aspect_dist.to_csv('aspect_distribution.csv', index=False)
        
        # Aspect Sentiment Scores
        aspect_sentiment_summary = df.groupby('aspect')['sentiment_score'].agg(['mean', 'count']).reset_index()
        aspect_sentiment_summary.columns = ['Aspect', 'Average_Sentiment', 'Review_Count']
        aspect_sentiment_summary.to_csv('aspect_sentiment_summary.csv', index=False)
        
        logging.info("CSV files created for sentiment analysis")

def main():
    # Configuration
    input_yaml_path = 'augmented_reviews.yaml'
    emotion_words_path = 'emotion_words.csv'
    aspect_keywords_path = 'aspect_keywords_merged.yaml'
    
    try:
        # Initialize analyzer
        analyzer = SentimentAnalyzer(
            emotion_words_path=emotion_words_path, 
            aspect_keywords_path=aspect_keywords_path
        )
        
        # Process reviews
        processed_reviews = analyzer.process_reviews(input_yaml_path)
        
        # Generate visualizations
        analyzer.generate_visualizations(processed_reviews)
        
        logging.info("Sentiment analysis complete. Check output files.")
    
    except Exception as e:
        logging.error(f"Fatal error during sentiment analysis: {e}")
        raise

if __name__ == "__main__":
    main()