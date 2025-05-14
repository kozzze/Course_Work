# train.py

import os
from transformers import Trainer, TrainingArguments

from spam_classifier.data import load_data, prepare_datasets, tokenize_function
from spam_classifier.model import get_model
from spam_classifier.utils import compute_metrics
from spam_classifier.config import DATA_PATH, SAVE_MODEL_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY


def main():
    # 1. Загрузка данных
    df = load_data(DATA_PATH)

    # 2. Подготовка датасетов
    train_dataset, val_dataset = prepare_datasets(df)

    # 3. Токенизация
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Удаляем оригинальные текстовые колонки (они больше не нужны)
    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])

    # 4. Загрузка модели
    model = get_model()

    # 5. Параметры тренировки
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 7. Обучение модели
    trainer.train()

    # 8. Сохранение модели
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    trainer.save_model(SAVE_MODEL_PATH)
    print(f"Модель успешно сохранена в {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    main()