# controller.py
from model import SpamModel

class SpamController:
    def __init__(self, model_path):
        self.model = SpamModel(model_path)

    def get_prediction(self, text):
        return self.model.predict(text)