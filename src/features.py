import numpy as np

LABEL_TO_ID = {
    "O": 0,
    "B-ANS": 1,
    "I-ANS": 2,
}


def token_features(question_tokens, context_tokens, index):
    word = context_tokens[index]

    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.prefix2": word[:2].lower(),
        "word.suffix2": word[-2:].lower(),
        "word.suffix3": word[-3:].lower(),
        "word.istitle": word.istitle(),
        "word.isupper": word.isupper(),
        "word.isdigit": word.isdigit(),
        "contains_hyphen": "-" in word,
        "contains_digit": any(char.isdigit() for char in word),
        "is_punctuation": all(not char.isalnum() for char in word),
        "relative_position": index / max(len(context_tokens), 1),
        "word_in_question": word.lower() in {q.lower() for q in question_tokens},
        "question_length": len(question_tokens),
        "question_has_digit": any(any(c.isdigit() for c in q) for q in question_tokens),
    }

    if index == 0:
        features["BOS"] = True
    else:
        previous = context_tokens[index - 1]

        features.update(
            {
                "previous.lower": previous.lower(),
                "previous.istitle": previous.istitle(),
                "previous.isupper": previous.isupper(),
                "previous.isdigit": previous.isdigit(),
            }
        )

    if index == len(context_tokens) - 1:
        features["EOS"] = True
    else:
        following = context_tokens[index + 1]

        features.update(
            {
                "next.lower": following.lower(),
                "next.istitle": following.istitle(),
                "next.isupper": following.isupper(),
                "next.isdigit": following.isdigit(),
            }
        )

    if index >= 2:
        features["previous2.lower"] = context_tokens[index - 2].lower()

    if index + 2 < len(context_tokens):
        features["next2.lower"] = context_tokens[index + 2].lower()

    for q_token in set(question_tokens):
        features[f"question_word={q_token.lower()}"] = True

    return features


def featurise_df(df):
    features = []
    labels = []
    lengths = []

    for _, row in df.iterrows():
        question_tokens = row["question_tokens"]
        context_tokens = row["context_tokens"]
        bio_labels = row["bio_labels"]

        lengths.append(len(context_tokens))

        for i in range(len(context_tokens)):
            features.append(
                token_features(
                    question_tokens,
                    context_tokens,
                    i,
                )
            )

            labels.append(LABEL_TO_ID[bio_labels[i]])

    return features, np.asarray(labels), lengths
