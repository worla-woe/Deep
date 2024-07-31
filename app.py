from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import nltk
import re
import string
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


# Load the vectorizer and model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define request model
class Request(BaseModel):
    text: str

# Function to transform the email text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
            
    return " ".join(y)

# Function to extract features
def extract_features(text):
    # Define custom features
    email_length = len(text)
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    average_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
    capital_letter_count = sum(1 for char in text if char.isupper())
    capital_letter_ratio = capital_letter_count / email_length if email_length > 0 else 0
    special_character_count = len(re.findall(r'[^a-zA-Z0-9\s]', text))


    # Keywords (can be adjusted based on your needs)
    keywords = [
    'free', 'winner', 'urgent', 'money', 'click', 'buy', 'discount', 'offer',
    'guarantee', 'limited', 'act now', 'exclusive', 'cash', 'prize', 'deal',
    'promotion', 'instant', 'win', 'bonus', 'trial', 'access', 'claim', 
    'investment', 'risk-free', 'savings', 'cancel at any time', 'hidden charges',
    'free gift', 'no cost', 'free quote', 'free membership', 'free access',
    'online biz opportunity', 'earn money', 'work from home', 'multi-level marketing',
    'make money', 'financial freedom', 'get paid', 'easy money', 'extra cash',
    'wealth', 'affiliate', 'best price', 'lowest price', 'free trial', 'free offer',
    'free info', 'free sample', '100% free', 'no fees', 'no hidden costs',
    'satisfaction guaranteed', 'free consultation', 'no purchase necessary',
    'limited time', 'special promotion', 'unsecured credit', 'credit repair',
    'debt relief', 'payday loan', 'cash bonus', 'insurance', 'mortgage',
    'consolidate debt', 'get paid', 'increase sales', 'online business',
    'search engine listings', 'social security number', 'eliminate bad credit',
    'get rich quick', 'unbelievable', 'unbelievable deal', 'free money', 
    'free rewards', 'free trial', 'lifetime', 'no strings attached', 
    'free download', 'free trial offer', 'free registration', 'free subscription',
    'free access', 'free membership', 'free gift card', 'free cash', 'free website',
    'free service', 'free trial subscription', 'free newsletter', 'free report',
    'free software', 'free app', 'free tips', 'free training', 'free resources',
    'free eBook', 'free guide', 'free course']
    keyword_presence = [1 if keyword in text.lower() else 0 for keyword in keywords]

    # Aggregate features into a single value (e.g., sum, mean, or any meaningful combination)
    combined_feature = (email_length + word_count + sentence_count +
                        punctuation_count + average_word_length + capital_letter_count +
                        special_character_count + sum(keyword_presence))  # Example of aggregation

    # Return a list with a single feature value
    return [combined_feature]


# Endpoint to make predictions
@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request):
    try:
        # Log incoming request
        print(f"Incoming request: {request}")

        # Preprocess the text
        processed_text = transform_text(request.text)
        
        # Extract features
        features = extract_features(processed_text)
        
        # Log features for debugging
        print(f"Extracted features: {features}")

        # Vectorize the transformed text
        transformed_text = vectorizer.transform([processed_text])
        
        # Log vectorized text shape for debugging
        print(f"Vectorized text shape: {transformed_text.shape}")

        # Ensure correct number of features
        expected_vector_length = 3200
        feature_array_length = expected_vector_length + len(features)

        if feature_array_length != 3201:
            raise ValueError(f"Feature dimension mismatch. Expected 3201 but got {feature_array_length}")

        # Combine transformed text with extracted features
        feature_array = np.hstack((transformed_text.toarray(), np.array(features).reshape(1, -1)))
        
        # Log combined feature array shape
        print(f"Feature array shape: {feature_array.shape}")

        # Make prediction using the model
        prediction_proba = model.predict_proba(feature_array)[0]
        prediction = model.predict(feature_array)[0]

        # Convert prediction to human-readable label
        label = "spam" if prediction == 1 else "ham"
        
        # Confidence level for the predicted class
        confidence = prediction_proba[prediction]
        
        # Return prediction and confidence as JSON response
        return {"prediction": label, "confidence": confidence}
    except HTTPException as e:
        raise e
    except ValueError as e:
        print(f"Value Error: {e}")
        raise HTTPException(status_code=400, detail=f"Feature dimension error: {e}")
    except Exception as e:
        # Log the exception for debugging
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")