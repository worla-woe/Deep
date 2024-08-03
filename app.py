from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import re
import string
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import logging

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    allow_origins=["https://phishing-detection-react-e95c9.web.app", "https://phishing-detection-react-e95c9.web.app/home", "https://phishing-detection-react-e95c9.web.app/predict","https://ella-161a4.firebaseapp.com/home","https://ella-161a4.web.app/home","https://ella-161a4.firebaseapp.com/predict","https://ella-161a4.web.app/predict","https://ella-161a4.web.app","https://ella-161a4.firebaseapp.com"],
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
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text =y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(lemmatizer.lemmatize(i))
            
    return " ".join(y)

# Function to extract features
def extract_features(text):
    email_length = len(text)
    word_count = len(text.split())
    punctuation_count = sum(1 for char in text if char in string.punctuation)

    # Calculate average word length safely
    average_word_length = (
        np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
    )

    capital_letter_count = sum(1 for char in text if char.isupper())
    special_character_count = len(re.findall(r'[^a-zA-Z0-9\s]', text))

    # Keywords (can be adjusted based on your needs)
    keywords = [
        'free', 'winner', 'urgent', 'money', 'click', 'buy',
        # ... (rest of your keywords)
    ]
    keyword_presence = [1 if keyword in text.lower() else 0 for keyword in keywords]

    # Aggregate features into a single value
    combined_feature = (
        email_length + word_count + punctuation_count +
        average_word_length + capital_letter_count +
        special_character_count + sum(keyword_presence)
    )

    return [combined_feature]

# Endpoint to make predictions
@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request):
    try:
        # Log incoming request
        logger.info(f"Incoming request: {request}")

        # Preprocess the text
        processed_text = transform_text(request.text)

        # Extract features
        features = extract_features(processed_text)

        # Log features for debugging
        logger.info(f"Extracted features: {features}")

        # Vectorize the transformed text
        transformed_text = vectorizer.transform([processed_text])

        # Ensure correct number of features
        expected_vector_length = 3200
        feature_array_length = expected_vector_length + len(features)

        if feature_array_length != 3201:
            raise ValueError(
                f"Feature dimension mismatch. Expected 3201 but got {feature_array_length}"
            )

        # Combine transformed text with extracted features
        feature_array = np.hstack((transformed_text.toarray(), np.array(features).reshape(1, -1)))

        # Make prediction using the model
        prediction_proba = model.predict_proba(feature_array)[0]
        prediction = model.predict(feature_array)[0]

        # Convert prediction to human-readable label
        label = "spam" if prediction == 1 else "ham"
        confidence = prediction_proba[prediction]

        # Return prediction and confidence as JSON response
        return {"prediction": label, "confidence": confidence}
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e}")
        raise e
    except ValueError as e:
        logger.error(f"Value Error: {e}")
        raise HTTPException(status_code=400, detail=f"Feature dimension error: {e}")
    except Exception as e:
        # Log the exception for debugging
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Root endpoint to verify the API is working
@app.get("/")
def read_root():
    return {"message": "Welcome to the PhishGuard API. Use the /predict endpoint to make predictions."}

@app.head("/")
async def head_root():
    return {}
