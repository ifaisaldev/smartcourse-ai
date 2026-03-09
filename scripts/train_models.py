import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_models():
    data_path = 'data/processed/courses.csv'
    model_dir = 'models'
    
    if not os.path.exists(data_path):
        print("Processed data not found. Please run prepare_data.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    # Fill NaNs in clean_description
    df['clean_description'] = df['clean_description'].fillna('')
    
    os.makedirs(model_dir, exist_ok=True)
    
    # --- TF-IDF Model ---
    print("Training TF-IDF Model...")
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['clean_description'])
    
    joblib.dump(tfidf, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(tfidf_matrix, os.path.join(model_dir, 'tfidf_matrix.pkl'))
    print("TF-IDF model saved.")
    
    # --- Neural Model ---
    print("Generating Neural Embeddings (this may take time)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Use raw description or clean? Usually raw is better for BERT-based models as they understand context,
    # but clean_description is also fine. Let's use description if available, else clean_description.
    # The prepare_data script ensures 'description' exists.
    
    texts = df['description'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    joblib.dump(embeddings, os.path.join(model_dir, 'neural_embeddings.pkl'))
    print("Neural embeddings saved.")

if __name__ == "__main__":
    train_models()
