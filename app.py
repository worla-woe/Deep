
import re
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import pickle
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
vectorizer_model = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize FastAPI
app = FastAPI()


@app.get("/hello")
async def read_hello():
    return {"msg": "hello"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://phishing-detection-react-e95c9.web.app",
                   "http://127.0.0.1:8000/predict"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request model
class PredictionRequest(BaseModel):
    text: str


# Function to transform the email text
def transform_text(message_text):
    text = message_text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(lemmatizer.lemmatize(i))
    return " ".join(y)


# Endpoint to make predictions
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Log incoming request
        logger.info(f"Incoming request: {request}")

        # Preprocess the text
        processed_text = transform_text(request.text)

        # Log features for debugging
        logger.info(f"Extracted features: {processed_text}")

        # Vectorize the transformed text
        transformed_text = vectorizer_model.transform([processed_text])
        # Extract the additional feature
        num_characters = len(request.text)
        num_words = len(request.text.split())
        num_special_chars = len(re.findall(
            r'[!@#$%^&*(),.?":{}|<>]', request.text))
        capital_ratio = sum(1 for c in request.text if c.isupper()
                            ) / num_characters if num_characters > 0 else 0
        avg_word_length = np.mean([len(word) for word in request.text.split()]
                                  ) if num_words > 0 else 0
        # Combine transformed text with the additional features
        feature_array = np.hstack((
            transformed_text.toarray(), np.array(
                [[num_characters, num_words, num_special_chars, capital_ratio,
                  avg_word_length,]])))

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
        raise HTTPException(
            status_code=400, detail=f"Feature dimension error: {e}")
    except Exception as e:
        # Log the exception for debugging
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Root endpoint to verify the API is working
@app.get("/")
def root_endpoint():
    return {'''"message": "Welcome to the PhishGuard API.
            Use the /predict endpoint to make predictions."'''}


@app.head("/")
async def head_root():
    return {}
