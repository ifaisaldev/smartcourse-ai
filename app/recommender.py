import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class CourseRecommender:
    def __init__(self, model_dir='models', data_path='data/processed/courses.csv'):
        self.model_dir = model_dir
        self.data_path = data_path
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.neural_embeddings = None
        self.neural_model = None
        
        self.load_data()
        self.load_models()

    def load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            # Ensure we have a clean_description column, fillna if needed
            self.df['clean_description'] = self.df['clean_description'].fillna('')
        else:
            print(f"Data file not found at {self.data_path}")

    def load_models(self):
        # Load TF-IDF
        tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        tfidf_matrix_path = os.path.join(self.model_dir, 'tfidf_matrix.pkl')
        
        if os.path.exists(tfidf_path) and os.path.exists(tfidf_matrix_path):
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            self.tfidf_matrix = joblib.load(tfidf_matrix_path)
        
        # Load Neural Embeddings
        neural_emb_path = os.path.join(self.model_dir, 'neural_embeddings.pkl')
        if os.path.exists(neural_emb_path):
            self.neural_embeddings = joblib.load(neural_emb_path)
            
        # Note: We load the sentence-transformer model only when needed or keep it in memory if frequent
        # For this implementation, we'll load it in the search method if needed, or here.
        # To save memory on startup, let's load it lazily or assume it's loaded by the training script.
        # Actually, for inference, we need the model to encode the query.
        try:
            from sentence_transformers import SentenceTransformer
            self.neural_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Could not load SentenceTransformer: {e}")

    def search_tfidf(self, query, top_k=10):
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        # Transform query
        from app.utils import clean_text
        clean_query = clean_text(query)
        query_vec = self.tfidf_vectorizer.transform([clean_query])
        
        # Calculate similarity
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = cosine_sim.argsort()[:-top_k-1:-1]
        
        results = []
        for idx in top_indices:
            if cosine_sim[idx] > 0: # Only return relevant results
                item = self.df.iloc[idx].to_dict()
                item['score'] = round(float(cosine_sim[idx]) * 100, 1)
                results.append(item)
        
        return results

    def search_neural(self, query, top_k=10):
        if self.neural_model is None or self.neural_embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.neural_model.encode([query])
        
        # Calculate similarity
        cosine_sim = cosine_similarity(query_embedding, self.neural_embeddings).flatten()
        
        # Get top k indices
        top_indices = cosine_sim.argsort()[:-top_k-1:-1]
        
        results = []
        for idx in top_indices:
            item = self.df.iloc[idx].to_dict()
            item['score'] = round(float(cosine_sim[idx]) * 100, 1)
            results.append(item)
            
        return results
