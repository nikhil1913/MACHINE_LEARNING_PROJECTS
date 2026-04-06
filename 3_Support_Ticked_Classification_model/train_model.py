import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

def setup():
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Stop words removal & lemmatization
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def main():
    setup()
    print("Loading dataset...")
    df = pd.read_csv('supportTicketData.csv')
    
    df_final = df.rename(columns={
        'Ticket detailed description': 'description',
        'urgency-Priority': 'priority'
    })
    
    if 'TicketID' in df_final.columns:
        df_final.drop('TicketID', axis=1, inplace=True)
        
    print("Cleaning text (this may take a minute)...")
    df_final['clean_text'] = df_final['description'].apply(clean_text)
    
    X = df_final['clean_text']
    y = df_final['priority']
    
    print("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X)
    
    print("Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    print("Training model...")
    # Using Logistic Regression as proposed
    model = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
    model.fit(X_tfidf, y_enc)
    
    print("Saving artifacts...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    print("Model serialized successfully in models/ folder!")

if __name__ == "__main__":
    main()
