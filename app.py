from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import nltk
import re
import string
from fastapi.middleware.cors import CORSMiddleware


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
    allow_origins=["http://localhost:5174"],  # Allow your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Define request model
class Request(BaseModel):
    text: str


# Function to transform the email text
def transform_text(emailText):
    text = emailText.lower()
    text = nltk.word_tokenize(text)
    filtered_words = [word for word in text if word.isalnum()]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return " ".join(word for word in filtered_words if word not in
                    stop_words and word not in string.punctuation)


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
    keyword_presence = [1 if keyword in text.lower() else 0 for
                        keyword in keywords]

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
        print(f"Incoming request: {request}")

        processed_text = transform_text(request.text)
        print(f"Processed text: {processed_text}")

        features = extract_features(processed_text)
        print(f"Extracted features: {features}")

        transformed_text = vectorizer.transform([processed_text])
        print(f"Transformed text shape: {transformed_text.shape}")

        expected_vector_length = 3200
        feature_array_length = expected_vector_length + len(features)

        if feature_array_length != 3201:
            raise ValueError(
                f"Feature dimension mismatch. Expected 3201 "
                f"but got {feature_array_length}"
            )

        feature_array = np.hstack((transformed_text.toarray(),
                                   np.array(features).reshape(1, -1)))
        print(f"Feature array shape: {feature_array.shape}")

        prediction_proba = model.predict_proba(feature_array)[0]
        prediction = model.predict(feature_array)[0]

        label = "spam" if prediction == 1 else "ham"
        confidence = prediction_proba[prediction]

        return {"prediction": label, "confidence": confidence}
    except HTTPException as e:
        print(f"HTTP Exception: {e.detail}")
        raise e
    except ValueError as e:
        print(f"Value Error: {e}")
        raise HTTPException(status_code=400, detail=f"Feature dimension error: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Root endpoint to verify the API is working
@app.get("/")
def read_root():
    return {"message": "Welcome to the PhishGuard API."
            'Use the /predict endpoint to make predictions.'}
