import pandas as pd


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
