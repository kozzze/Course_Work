# data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset

from .config import MODEL_NAME, MAX_LENGTH, TRAIN_TEST_SPLIT

def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df[['label', 'text']]
    df['text'] = df['text'].astype(str)
    return df

def prepare_datasets(df):

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        train_size=TRAIN_TEST_SPLIT,
        random_state=42,
        stratify=df['label']
    )

    # Обернем в Dataset
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })

    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })

    return train_dataset, val_dataset

def tokenize_function(examples):
    """Токенизация данных"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )