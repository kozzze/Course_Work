# utils.py

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1
    }