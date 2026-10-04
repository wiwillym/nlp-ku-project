import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


def evaluate_answerability_predictions(y_true, y_pred, languages, model_name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    languages = np.asarray(languages)

    groups = {
        "Overall": np.ones(len(y_true), dtype=bool),
        "Arabic": languages == "ar",
        "Korean": languages == "ko",
        "Telugu": languages == "te",
    }

    rows = []

    for language, mask in groups.items():
        gold = y_true[mask]
        pred = y_pred[mask]

        precision, recall, f1, _ = precision_recall_fscore_support(
            gold,
            pred,
            labels=[0, 1],
            zero_division=0,
        )

        rows.append(
            {
                "Model": model_name,
                "Language": language,
                "N": len(gold),
                "Macro F1": f1_score(gold, pred, average="macro"),
                "Unanswerable Precision": precision[0],
                "Unanswerable Recall": recall[0],
                "Unanswerable F1": f1[0],
                "Answerable Precision": precision[1],
                "Answerable Recall": recall[1],
                "Answerable F1": f1[1],
            }
        )

    return pd.DataFrame(rows)
