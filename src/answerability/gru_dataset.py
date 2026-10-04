import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PretrainedAnswerabilityDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_question_length=64,
        max_context_length=448,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        question = " ".join(list(row["question_tokens"]))
        context = " ".join(list(row["context_tokens"]))

        question_encoding = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_question_length,
            add_special_tokens=True,
        )

        context_encoding = self.tokenizer(
            context,
            truncation=True,
            max_length=self.max_context_length,
            add_special_tokens=True,
        )

        return {
            "question_ids": torch.tensor(
                question_encoding["input_ids"],
                dtype=torch.long,
            ),
            "context_ids": torch.tensor(
                context_encoding["input_ids"],
                dtype=torch.long,
            ),
            "label": torch.tensor(
                int(row["answerable"]),
                dtype=torch.float,
            ),
            "lang": row["lang"],
        }


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        question_ids = pad_sequence(
            [item["question_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )

        context_ids = pad_sequence(
            [item["context_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )

        question_lengths = torch.tensor(
            [len(item["question_ids"]) for item in batch],
            dtype=torch.long,
        )

        context_lengths = torch.tensor(
            [len(item["context_ids"]) for item in batch],
            dtype=torch.long,
        )

        labels = torch.stack([item["label"] for item in batch])

        langs = [item["lang"] for item in batch]

        return {
            "question_ids": question_ids,
            "question_lengths": question_lengths,
            "context_ids": context_ids,
            "context_lengths": context_lengths,
            "labels": labels,
            "lang": langs,
        }

    return collate_fn
