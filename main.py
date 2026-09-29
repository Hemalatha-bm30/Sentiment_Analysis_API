#1. IMPORTS

import re  #cleans incoming text
import joblib #loads our saved .pkl files

from fastapi import FastAPI #creates the API
from pydantic import BaseModel #defines the format of data the API receives

#2 — Load the saved model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

#3 — Create the FastAPI app
app = FastAPI()

#4 — Define what the API receives
class TextInput(BaseModel):
    text: str

#5 — Create the prediction endpoint
@app.post("/predict")
def predict_sentiment(data: TextInput):

    text = data.text

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)[0]

    return {
        "text": data.text,
        "sentiment": prediction
    }