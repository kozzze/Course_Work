# model.py

from transformers import AutoModelForSequenceClassification
from .config import MODEL_NAME

def get_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # Два класса: спам и не спам
    )
    return model