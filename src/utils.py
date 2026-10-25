import torch
import random
import numpy as np
import pandas as pd

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


def split_by_lengths(values, lengths):
    sequences = []
    offset = 0

    for length in lengths:
        sequences.append(list(values[offset : offset + length]))
        offset += length

    assert offset == len(values)

    return sequences


def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)
