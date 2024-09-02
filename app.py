from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import string
import logging
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

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


# Load the deployment pipeline and model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


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


class PredictionRequest(BaseModel):
    text: str
   

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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Log incoming request
        logger.info(f"Incoming request: {request}")

        # Preprocess the text
        processed_text = transform_text(request.text)

        # Log features for debugging
        logger.info(f"Extracted features: {processed_text}")

        # Use the model pipeline to predict
        prediction_proba = model.predict_proba([processed_text])[0]
        prediction = model.predict([request.text])[0]
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
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/")
def root_endpoint():
    return {"message": "PhishGuard! Use the /predict endpoint for prediction."}


@app.head("/")
async def head_root():
    return {}
