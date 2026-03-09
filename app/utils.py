import spacy
import re

# Load spaCy model (ensure it's downloaded via 'python -m spacy download en_core_web_sm')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Cleans and preprocesses text:
    1. Lowercase
    2. Remove special characters/HTML
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags (basic regex)
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Process with spaCy
    doc = nlp(text)
    
    # 4 & 5. Remove stopwords and Lemmatize
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    
    return " ".join(tokens)
