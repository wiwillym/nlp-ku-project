from collections import defaultdict
import pandas as pd
import numpy as np


def evaluate_sequences(
    pred_sequences,
    gold_sequences,
    languages,
    model_name,
):
    stats = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact": 0,
            "n": 0,
        }
    )

    for pred, gold, lang in zip(
        pred_sequences,
        gold_sequences,
        languages,
    ):
        pred = np.asarray(pred)
        gold = np.asarray(gold)

        # B and I are both answer tokens
        pred_answer = pred != 0
        gold_answer = gold != 0

        tp = np.sum(pred_answer & gold_answer)
        fp = np.sum(pred_answer & ~gold_answer)
        fn = np.sum(~pred_answer & gold_answer)

        exact = np.array_equal(
            pred_answer,
            gold_answer,
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

        precision = (
            s["tp"] / (s["tp"] + s["fp"])
            if s["tp"] + s["fp"] > 0
            else 0
        )

        recall = (
            s["tp"] / (s["tp"] + s["fn"])
            if s["tp"] + s["fn"] > 0
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        exact_match = (
            s["exact"] / s["n"]
            if s["n"] > 0
            else 0
        )

        rows.append({
            "Model": model_name,
            "Language": lang_names[lang],
            "N": s["n"],
            "Precision": precision,
            "Recall": recall,
            "Token F1": f1,
            "Exact Span Match": exact_match,
        })

    return pd.DataFrame(rows)
