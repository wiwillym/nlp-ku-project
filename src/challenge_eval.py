"""Evaluation helpers for the paired challenge set (Weeks 43-44).

Every model produces one row per challenge record with at least
    id, pair_id, lang, answerable (gold, bool), pred_answerable (bool)
and, for span-producing models, `em` and `f1` (floats, only meaningful on
answerable records).

Metrics are reported separately for the answerable and the unanswerable set:
  * answerable set:   answered rate (= answerable recall), and EM / token F1
  * unanswerable set: abstention rate (= unanswerable recall); for span models
                      this equals the share of empty predictions, i.e. EM
Pair level: both records of a pair handled correctly.

The main table averages the per-language values over languages; the same
function with by_language=True gives the appendix table.
"""

from collections import defaultdict

import numpy as np
import pandas as pd


def get_span_scores(
    model,
    tokenizer,
    df,
    device,
    max_length=384,
    doc_stride=128,
    batch_size=32,
    num_possible_answers=20,
    max_answer_length=30,
):
    
    import torch
    from torch.utils.data import DataLoader
    from src.integrated_qa import build_qa_features, val_collate_fn

    df = df.reset_index(drop=True)
    features = build_qa_features(df, tokenizer, max_length=max_length, doc_stride=doc_stride)
    prediction_features = [
        {
            "input_ids": f["input_ids"],
            "attention_mask": f["attention_mask"],
            "example_index": f["example_index"],
            "offset_mapping": [
                offset if sid == 1 else None
                for offset, sid in zip(f["offset_mapping"], f["sequence_ids"])
            ],
        }
        for f in features
    ]
    loader = DataLoader(prediction_features, collate_fn=val_collate_fn, batch_size=batch_size, shuffle=False)

    model.eval()
    all_start, all_end = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_start.extend(out.start_logits.cpu().numpy())
            all_end.extend(out.end_logits.cpu().numpy())

    by_example = defaultdict(list)
    for i, f in enumerate(prediction_features):
        by_example[f["example_index"]].append(i)

    rows = []
    for idx, context in enumerate(df["context"].tolist()):
        best_span, best_text, best_null = -np.inf, "", np.inf
        for fi in by_example[idx]:
            start_logits, end_logits = all_start[fi], all_end[fi]
            offsets = prediction_features[fi]["offset_mapping"]
            best_null = min(best_null, start_logits[0] + end_logits[0])
            for s in np.argsort(start_logits)[::-1][:num_possible_answers]:
                for e in np.argsort(end_logits)[::-1][:num_possible_answers]:
                    if s >= len(offsets) or e >= len(offsets) or offsets[s] is None or offsets[e] is None:
                        continue
                    if s > e or e - s + 1 > max_answer_length:
                        continue
                    score = start_logits[s] + end_logits[e]
                    if score > best_span:
                        best_span = score
                        best_text = context[offsets[s][0] : offsets[e][1]]
        rows.append({"span_text": best_text, "no_answer_score": float(best_null - best_span)})
    return pd.DataFrame(rows)


def apply_threshold(scores, langs, threshold):
    t = (
        np.array([threshold[l] for l in langs])
        if isinstance(threshold, dict)
        else np.full(len(scores), float(threshold))
    )
    return np.where(scores["no_answer_score"].to_numpy() > t, "", scores["span_text"].to_numpy())


def qa_result_rows(challenge_df, predictions):
    from src.integrated_qa import exact_match_score, f1_score_squad

    rows = challenge_df[["id", "pair_id", "lang", "answerable"]].copy().reset_index(drop=True)
    rows["answerable"] = rows["answerable"].astype(bool)
    rows["prediction"] = list(predictions)
    gold = challenge_df["answer"].tolist()
    rows["em"] = [float(exact_match_score(p, g)) for p, g in zip(predictions, gold)]
    rows["f1"] = [float(f1_score_squad(p, g)) for p, g in zip(predictions, gold)]
    rows["pred_answerable"] = [len(p) > 0 for p in predictions]
    return rows


def classifier_result_rows(challenge_df, pred_answerable):
    rows = challenge_df[["id", "pair_id", "lang", "answerable"]].copy().reset_index(drop=True)
    rows["answerable"] = rows["answerable"].astype(bool)
    rows["pred_answerable"] = np.asarray(pred_answerable).astype(bool)
    return rows


def span_labeller_result_rows(challenge_df, pred_labels, gold_labels):
    rows = challenge_df[["id", "pair_id", "lang", "answerable"]].copy().reset_index(drop=True)
    rows["answerable"] = rows["answerable"].astype(bool)
    ems, f1s, answered = [], [], []
    for pred, gold in zip(pred_labels, gold_labels):
        p = set(np.flatnonzero(np.asarray(pred) != 0))
        g = set(np.flatnonzero(np.asarray(gold) != 0))
        ems.append(float(p == g))
        if not p and not g:
            f1s.append(1.0)
        elif not p or not g:
            f1s.append(0.0)
        else:
            tp = len(p & g)
            f1s.append(0.0 if tp == 0 else 2 * tp / (len(p) + len(g)))
        answered.append(len(p) > 0)
    rows["em"], rows["f1"], rows["pred_answerable"] = ems, f1s, answered
    return rows


def _summarise_language(res):
    ans = res[res["answerable"]]
    unans = res[~res["answerable"]]
    out = {
        "n_answerable": len(ans),
        "answered": ans["pred_answerable"].mean() if len(ans) else np.nan,
        "n_unanswerable": len(unans),
        "abstained": (~unans["pred_answerable"]).mean() if len(unans) else np.nan,
    }
    if "em" in res:
        out["em"] = ans["em"].mean() if len(ans) else np.nan
        out["f1"] = ans["f1"].mean() if len(ans) else np.nan
    # pairs: answerable record answered and unanswerable record abstained
    pair_ok, pair_exact = [], []
    for _, g in res.groupby("pair_id"):
        a, u = g[g["answerable"]], g[~g["answerable"]]
        if len(a) != 1 or len(u) != 1:
            continue
        ok = bool(a["pred_answerable"].iloc[0]) and not bool(u["pred_answerable"].iloc[0])
        pair_ok.append(ok)
        if "em" in res:
            pair_exact.append(ok and a["em"].iloc[0] == 1.0)
    out["n_pairs"] = len(pair_ok)
    out["pairs_correct"] = float(np.mean(pair_ok)) if pair_ok else np.nan
    if "em" in res:
        out["pairs_exact"] = float(np.mean(pair_exact)) if pair_exact else np.nan
    return out


def summarise_challenge(results_by_model, by_language=False):
    rows = []
    for model, res in results_by_model.items():
        per_lang = {lang: _summarise_language(g) for lang, g in res.groupby("lang")}
        if by_language:
            for lang, s in per_lang.items():
                rows.append({"model": model, "lang": lang, **s})
        else:
            frame = pd.DataFrame(per_lang).T
            avg = frame.mean(numeric_only=True)
            for count in ["n_answerable", "n_unanswerable", "n_pairs"]:
                avg[count] = frame[count].sum()
            rows.append({"model": model, **avg.to_dict()})
    return pd.DataFrame(rows).round(3)
