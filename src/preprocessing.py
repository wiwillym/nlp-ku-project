import os
import stanza
import pandas as pd
from data import load_data, DATA_PATH

LANGUAGES_TO_TOKENISE = ["ar", "ko", "te"]

tokenisers = {
    lang: stanza.Pipeline(
        lang=lang,
        processors="tokenize",
        verbose=False,
    )
    for lang in LANGUAGES_TO_TOKENISE + ["en"]
}


def tokenise(text, lang):
    doc = tokenisers[lang](text)

    return [token.text for sentence in doc.sentences for token in sentence.tokens]


def tokenise_context_and_answer(context, answer, lang):
    context_tokens = tokenise(context, lang)
    answer_tokens = tokenise(answer, lang)
    return context_tokens, answer_tokens


def tokenise_dataset(df):
    df = df[df["lang"].isin(LANGUAGES_TO_TOKENISE)]
    tokenised_datasets = []
    for lang in LANGUAGES_TO_TOKENISE:
        print(f"Tokenising {lang} dataset...")
        sub_df_by_lang = df[df["lang"] == lang]
        sub_df_by_lang["context_tokens"], sub_df_by_lang["answer_tokens"] = zip(
            *sub_df_by_lang.apply(
                lambda row: tokenise_context_and_answer(
                    row["context"], row["answer"], "en"
                ),
                axis=1,
            )
        )
        sub_df_by_lang["question_tokens"] = sub_df_by_lang["question"].apply(
            lambda x: tokenise(x, lang)
        )
        tokenised_datasets.append(sub_df_by_lang)
    df = pd.concat(tokenised_datasets, ignore_index=True)
    return df


def tokenise_train_and_val():
    df_train, df_val = load_data()
    df_train_tokenised = tokenise_dataset(df_train)
    df_val_tokenised = tokenise_dataset(df_val)
    df_train_tokenised.to_parquet(
        os.path.join(DATA_PATH, "train_tokenised.parquet"), index=False
    )
    df_val_tokenised.to_parquet(
        os.path.join(DATA_PATH, "validation_tokenised.parquet"), index=False
    )
    return df_train_tokenised, df_val_tokenised


if __name__ == "__main__":
    tokenise_train_and_val()
