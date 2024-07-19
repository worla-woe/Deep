from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load the vectorizer and model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize FastAPI
app = FastAPI()


# Define request model
class Message(BaseModel):
    text: str


# Function to preprocess text
def preprocess_text(text: str) -> str:
    # Example: Remove leading and trailing whitespace
    return text.strip()


# Function to extract the number of characters
def get_num_characters(text: str) -> int:
    return len(text)


# Endpoint to make predictions
@app.post("/predict", response_model=dict)
def predict(message: Message):
    try:
        # Log incoming request
        print(f"Incoming request: {message}")

        # Preprocess the text
        processed_text = preprocess_text(message.text)

        # Extract the number of characters
        num_characters = get_num_characters(processed_text)
        # Transform the input text using the vectorizer
        transformed_text = vectorizer.transform([processed_text])
        # Combine transformed text with the num_characters feature
        transformed_text_with_num_chars = np.hstack((
            transformed_text.toarray(), [[num_characters]]))
        # Make prediction using the model
        prediction = model.predict(transformed_text_with_num_chars)
        # Convert prediction to human-readable label
        label = "spam" if prediction[0] == 1 else "ham"
        # Return prediction as JSON response
        return {"prediction": label}
    except HTTPException as e:
        raise e
    except Exception as e:
        # Log the exception for debugging
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Endpoint to check the API status
@app.get("/")
def read_root():
    return {"message": "Spam detection API is up and running!"}


