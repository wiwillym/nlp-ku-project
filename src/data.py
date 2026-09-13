import os
from pathlib import Path
import sys

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_PATH = os.path.join(PROJECT_ROOT, "data")

TRAIN_PATH = os.path.join(DATA_PATH, "train.parquet")
VAL_PATH = os.path.join(DATA_PATH, "validation.parquet")
TRAIN_TOKENISED_PATH = os.path.join(DATA_PATH, "train_tokenised.parquet")
VAL_TOKENISED_PATH = os.path.join(DATA_PATH, "validation_tokenised.parquet")


def load_data():
    os.makedirs(DATA_PATH, exist_ok=True)

    if os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH):
        df_train = pd.read_parquet(TRAIN_PATH)
        df_val = pd.read_parquet(VAL_PATH)

    else:
        dataset = load_dataset("coastalcph/tydi_xor_rc")

        df_train = dataset["train"].to_pandas()
        df_val = dataset["validation"].to_pandas()

        df_train.to_parquet(TRAIN_PATH, index=False)
        df_val.to_parquet(VAL_PATH, index=False)

    return df_train, df_val


def load_tokenised_data():
    os.makedirs(DATA_PATH, exist_ok=True)

    if os.path.exists(TRAIN_TOKENISED_PATH) and os.path.exists(VAL_TOKENISED_PATH):
        df_train_tokenised = pd.read_parquet(TRAIN_TOKENISED_PATH)
        df_val_tokenised = pd.read_parquet(VAL_TOKENISED_PATH)
    else:
        raise FileNotFoundError("Tokenised data not found.")

    return df_train_tokenised, df_val_tokenised


if __name__ == "__main__":
    df_train, df_val = load_data()
