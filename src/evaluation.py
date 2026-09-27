import torch
import pandas as pd
from collections import defaultdict
from src.labeller import LocalSequenceLabeller

LABEL_TO_ID = {
    "O": 0,
    "B-ANS": 1,
    "I-ANS": 2,
}


def tokens_to_ids(tokens, vocab):
    return [vocab.get(token.lower(), vocab["<UNK>"]) for token in tokens]


def evaluate_model(model, dataloader, device, model_name):
    stats = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact": 0,
            "n": 0,
        }
    )

    def predict_batch(model, batch, device):
        # Logistic Regression
        if isinstance(model, LocalSequenceLabeller):
            return model.predict_batch(batch)

        model.eval()

        question_ids = batch["question_ids"].to(device)
        context_ids = batch["context_ids"].to(device)

        with torch.no_grad():
            logits = model(
                question_ids,
                context_ids,
            )

        predictions = logits.argmax(dim=-1)

        return predictions

    for batch in dataloader:
        predictions = predict_batch(
            model,
            batch,
            device,
        )

        labels = batch["labels"]
        langs = batch["lang"]

        for i, (gold, lang) in enumerate(zip(labels, langs)):
            valid = gold != -100
            gold = gold[valid]

            if isinstance(predictions, list):
                pred = predictions[i]
            else:
                pred = predictions[i].cpu()
                pred = pred[valid]

            true_answer = gold != 0
            pred_answer = pred != 0

            tp = (true_answer & pred_answer).sum().item()

            fp = (~true_answer & pred_answer).sum().item()

            fn = (true_answer & ~pred_answer).sum().item()

            exact = torch.equal(
                true_answer,
                pred_answer,
            )

            for key in ["overall", lang]:
                stats[key]["tp"] += tp
                stats[key]["fp"] += fp
                stats[key]["fn"] += fn
                stats[key]["exact"] += int(exact)
                stats[key]["n"] += 1

    lang_names = {
        "overall": "Overall",
        "ar": "Arabic",
        "ko": "Korean",
        "te": "Telugu",
    }

    rows = []

    for lang in ["overall", "ar", "ko", "te"]:
        s = stats[lang]

        precision = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] > 0 else 0

        recall = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        exact_match = s["exact"] / s["n"] if s["n"] > 0 else 0

        rows.append(
            {
                "Model": model_name,
                "Language": lang_names[lang],
                "N": s["n"],
                "Precision": precision,
                "Recall": recall,
                "Token F1": f1,
                "Exact Span Match": exact_match,
            }
        )

    return pd.DataFrame(rows)
