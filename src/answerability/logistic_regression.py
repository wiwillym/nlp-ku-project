import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


def answerability_features(row):
    question = [token.lower() for token in row["question_tokens"]]
    context = [token.lower() for token in row["context_tokens"]]

    question_set = set(question)
    context_set = set(context)
    overlap = question_set & context_set

    features = {
        "question_length": len(question),
        "context_length": len(context),
        "overlap_count": len(overlap),
        "overlap_question_ratio": (
            len(overlap) / len(question_set) if len(question_set) > 0 else 0.0
        ),
        "question_has_digit": any(
            any(char.isdigit() for char in token) for token in question
        ),
        "context_has_digit": any(
            any(char.isdigit() for char in token) for token in context
        ),
    }

    for token in question_set:
        features[f"q_word={token}"] = 1

    for token in context_set:
        features[f"c_word={token}"] = 1

    return features


def train_logistic_regression(train_df):
    train_features = [answerability_features(row) for _, row in train_df.iterrows()]
    y_train = train_df["answerable"].astype(int).to_numpy()

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_features)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    return model, vectorizer


def predict_logistic_regression(model, vectorizer, df):
    features = [answerability_features(row) for _, row in df.iterrows()]

    X = vectorizer.transform(features)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities


def logistic_feature_weights(model, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "weight": weights,
        }
    )

    return result.sort_values("weight").reset_index(drop=True)
