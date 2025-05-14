# model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SpamModel:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        return "Спам" if prediction == 1 else "Не спам"