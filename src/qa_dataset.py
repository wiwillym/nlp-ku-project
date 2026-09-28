import torch
from torch.utils.data import Dataset

LABEL_TO_ID = {
    "O": 0,
    "B-ANS": 1,
    "I-ANS": 2,
}

def tokens_to_ids(tokens, vocab):
    return [
        vocab.get(token.lower(), vocab["<UNK>"])
        for token in tokens
    ]

class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        question_ids = tokens_to_ids(
            row["question_tokens"],
            self.vocab,
        )

        context_ids = tokens_to_ids(
            row["context_tokens"],
            self.vocab,
        )

        labels = [
            LABEL_TO_ID[label]
            for label in row["bio_labels"]
        ]

        return {
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "context_ids": torch.tensor(context_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "lang": row["lang"],
        }
