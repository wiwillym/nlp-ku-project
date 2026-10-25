"""Calibrated abstention for the integrated QA model (Week 43 extension).

The QA model gives every example a no-answer score (null score minus best span
score). Higher means "more likely unanswerable"; the example is answered when
score <= threshold. Core model: one global threshold. Extension: compare it with
per-language thresholds, and measure whether language-specific Platt scaling
gives better-calibrated probabilities.

Every method is fitted and scored with the same repeated, stratified k-fold
cross-validation *inside* the validation set, so no number is tuned and reported
on the same examples.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

LANGS = ["ar", "ko", "te"]
GROUPS = ["Overall"] + LANGS


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def best_threshold(scores, y):
    """Threshold t maximising macro-F1 when predicting answerable iff score <= t.

    y is 1 for answerable. Ties between equally good cut points are broken by
    taking the middle one, which is more stable than the first.
    """
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    order = np.argsort(scores, kind="stable")
    s, yy = scores[order], y[order]
    n = len(s)
    n_ans = yy.sum()
    n_un = n - n_ans

    # k lowest-scoring examples are predicted answerable, k = 0..n
    k = np.arange(n + 1)
    tp_a = np.concatenate([[0], np.cumsum(yy)])
    fp_a = k - tp_a
    fn_a = n_ans - tp_a
    tp_u = n_un - fp_a
    macro = (_f1(tp_a, fp_a, fn_a) + _f1(tp_u, fn_a, fp_a)) / 2

    valid = np.ones(n + 1, bool)
    valid[1:n] = s[1:] != s[:-1]  # cannot cut between equal scores
    macro = np.where(valid, macro, -1.0)

    best = macro.max()
    candidates = np.flatnonzero(macro >= best - 1e-12)
    cut = candidates[len(candidates) // 2]
    if cut == 0:
        threshold = s[0] - 1.0
    elif cut == n:
        threshold = s[-1] + 1.0
    else:
        threshold = (s[cut - 1] + s[cut]) / 2
    return float(threshold), float(best)


class GlobalThreshold:
    name = "global"

    def fit(self, scores, y, langs):
        self.threshold, _ = best_threshold(scores, y)
        return self

    def predict(self, scores, langs):
        return (np.asarray(scores, float) <= self.threshold).astype(int)


class PerLanguageThreshold:
    """One threshold per language, falling back to the global one when a
    language has fewer than `min_per_class` examples of either class."""

    name = "per_language"

    def __init__(self, min_per_class=5):
        self.min_per_class = min_per_class

    def fit(self, scores, y, langs):
        scores, y, langs = np.asarray(scores, float), np.asarray(y, int), np.asarray(langs)
        self.global_threshold, _ = best_threshold(scores, y)
        self.thresholds = {}
        for lang in np.unique(langs):
            mask = langs == lang
            n_ans = int(y[mask].sum())
            if min(n_ans, int(mask.sum()) - n_ans) >= self.min_per_class:
                self.thresholds[lang], _ = best_threshold(scores[mask], y[mask])
            else:
                self.thresholds[lang] = self.global_threshold
        return self

    def predict(self, scores, langs):
        t = np.array([self.thresholds.get(l, self.global_threshold) for l in langs])
        return (np.asarray(scores, float) <= t).astype(int)


class Platt:
    """P(answerable) = sigmoid(a * (-score) + b), optionally with a
    language-specific intercept b_lang (shared slope a)."""

    def __init__(self, per_language):
        self.per_language = per_language
        self.name = "platt_per_language" if per_language else "platt_global"

    def _features(self, scores, langs):
        cols = [-np.asarray(scores, float)]
        if self.per_language:
            langs = np.asarray(langs)
            cols += [(langs == l).astype(float) for l in LANGS[1:]]
        return np.column_stack(cols)

    def fit(self, scores, y, langs):
        self.model = LogisticRegression(C=1e6, max_iter=2000)
        self.model.fit(self._features(scores, langs), np.asarray(y, int))
        return self

    def predict_proba(self, scores, langs):
        return self.model.predict_proba(self._features(scores, langs))[:, 1]


def expected_calibration_error(prob, y, n_bins=10):
    prob, y = np.asarray(prob, float), np.asarray(y, float)
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(ece)


def _classification_metrics(y, pred):
    out = {}
    tp_a = int(((pred == 1) & (y == 1)).sum())
    fp_a = int(((pred == 1) & (y == 0)).sum())
    fn_a = int(((pred == 0) & (y == 1)).sum())
    tp_u = int(((pred == 0) & (y == 0)).sum())
    f1_a = float(_f1(tp_a, fp_a, fn_a))
    f1_u = float(_f1(tp_u, fn_a, fp_a))
    out["macro_f1"] = (f1_a + f1_u) / 2
    out["unans_precision"] = tp_u / (tp_u + fn_a) if tp_u + fn_a else 0.0
    out["unans_recall"] = tp_u / (tp_u + fp_a) if tp_u + fp_a else 0.0
    out["unans_f1"] = f1_u
    out["ans_recall"] = tp_a / (tp_a + fn_a) if tp_a + fn_a else 0.0
    return out


def cross_validate(df, n_splits=5, n_repeats=5, seed=42, min_per_class=5):
    """Repeated stratified CV over the validation set.

    df needs columns: lang, gold_answerable, no_answer_score. If it also has
    `prediction` (best span text) and `gold_answer`, SQuAD-v2 EM and token F1 of
    the abstaining system are reported too.

    Returns a long DataFrame: one row per repeat x method x group x metric
    value, using out-of-fold predictions pooled within each repeat.
    """
    scores = df["no_answer_score"].to_numpy(float)
    y = df["gold_answerable"].to_numpy(int)
    langs = df["lang"].to_numpy()
    has_text = {"prediction", "gold_answer"} <= set(df.columns)
    strata = np.char.add(langs.astype(str), y.astype(str))

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    n = len(df)
    decisions = [GlobalThreshold, lambda: PerLanguageThreshold(min_per_class)]
    calibrators = [lambda: Platt(False), lambda: Platt(True)]

    oof = {}  # (repeat, method) -> predictions / probabilities
    for i, (train, test) in enumerate(splitter.split(scores, strata)):
        repeat = i // n_splits
        for make in decisions:
            method = make().fit(scores[train], y[train], langs[train])
            oof.setdefault((repeat, method.name), np.zeros(n, int))[test] = method.predict(
                scores[test], langs[test]
            )
        for make in calibrators:
            method = make().fit(scores[train], y[train], langs[train])
            oof.setdefault((repeat, method.name), np.zeros(n))[test] = method.predict_proba(
                scores[test], langs[test]
            )

    if has_text:
        from src.integrated_qa import exact_match_score, f1_score_squad

        texts = df["prediction"].fillna("").astype(str).to_numpy()
        golds = df["gold_answer"].fillna("").astype(str).to_numpy()

    rows = []
    for (repeat, name), values in oof.items():
        for group in GROUPS:
            mask = np.ones(n, bool) if group == "Overall" else langs == group
            row = {"repeat": repeat, "method": name, "group": group, "n": int(mask.sum())}
            if name.startswith("platt"):
                row["ece"] = expected_calibration_error(values[mask], y[mask])
                row["brier"] = float(np.mean((values[mask] - y[mask]) ** 2))
            else:
                row.update(_classification_metrics(y[mask], values[mask]))
                if has_text:
                    final = np.where(values == 1, texts, "")
                    idx = np.flatnonzero(mask)
                    row["squad_em"] = float(np.mean([exact_match_score(final[j], golds[j]) for j in idx]))
                    row["squad_f1"] = float(np.mean([f1_score_squad(final[j], golds[j]) for j in idx]))
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(results, metrics):
    """Mean and standard deviation over repeats."""
    return (
        results.groupby(["method", "group"])[metrics]
        .agg(["mean", "std"])
        .round(4)
    )


def paired_difference(results, method_a, method_b, metric):
    """method_a - method_b per repeat and group: mean, std, and the number of
    repeats in which method_a is strictly better."""
    wide = results.pivot_table(index=["repeat", "group"], columns="method", values=metric)
    diff = (wide[method_a] - wide[method_b]).reset_index(name="diff")
    return (
        diff.groupby("group")["diff"]
        .agg(mean="mean", std="std", repeats_better=lambda d: int((d > 1e-12).sum()), repeats="size")
        .round(4)
    )


def evaluate_at_threshold(df, threshold):
    """In-sample metrics at a fixed threshold (e.g. the one the QA notebook tuned
    on all of validation). Optimistic by construction; for reference only."""
    y = df["gold_answerable"].to_numpy(int)
    pred = (df["no_answer_score"].to_numpy(float) <= threshold).astype(int)
    langs = df["lang"].to_numpy()
    rows = []
    for group in GROUPS:
        mask = np.ones(len(df), bool) if group == "Overall" else langs == group
        rows.append({"group": group, "n": int(mask.sum()), **_classification_metrics(y[mask], pred[mask])})
    return pd.DataFrame(rows).round(4)
