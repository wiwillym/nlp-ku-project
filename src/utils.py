import pandas as pd
from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def get_most_common_tokens(df, translator, n=5):
    rows = []

    for lang in df["lang"].unique():

        most_common_tokens = (
            df.loc[df["lang"] == lang, "question_tokens"]
            .explode()
            .value_counts()
            .head(n)
        )

        for token, count in most_common_tokens.items():

            translation = translator.translate(
                token,
                source_lang=lang,
            )

            rows.append(
                {
                    "language": lang,
                    "token": token,
                    "count": int(count),
                    "translation": translation,
                }
            )

    return pd.DataFrame(rows)


def build_vocabulary(df):
    counter = Counter()
    for tokens in df["context_tokens"]:
        counter.update(token.lower() for token in tokens)

    for tokens in df["question_tokens"]:
        counter.update(token.lower() for token in tokens)

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, count in counter.items():
        if count >= 10:
            vocabulary[token] = len(vocabulary)

    return vocabulary
