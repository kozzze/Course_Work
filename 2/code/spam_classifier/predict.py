from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Путь к сохранённой модели
model_path = "/Users/kozzze/Desktop/Учеба/Course_Work/2/code/model"

# Загрузка модели и токенизатора
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")


def predict(text):
    # Токенизация
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)

    # Прогон модели
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Получаем предсказание (0 - не спам, 1 - спам)
    prediction = torch.argmax(logits, dim=-1).item()

    if prediction == 1:
        return "Спам"
    else:
        return "Не спам"


# Пример:
text = "привет! как твои дела? видел этот прикол https://tiktok/video-12345?"
result = predict(text)
print(f"Сообщение: {text}\nРезультат: {result}")