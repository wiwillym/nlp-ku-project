import re
import string
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def build_qa_features(df, tokenizer, max_length=384, doc_stride=128):
    samples = df.reset_index(drop=True)

    batch = tokenizer(
        samples["question"].tolist(),
        samples["context"].tolist(),
        padding="max_length",
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    sample_mapping = batch.pop("overflow_to_sample_mapping")

    offset_mapping = batch.pop("offset_mapping")

    batch["start_positions"] = []
    batch["end_positions"] = []

    # We keep this information for validation/evaluation later
    batch["example_index"] = []
    batch["offset_mapping"] = []
    batch["sequence_ids"] = []

    for i, offsets in enumerate(offset_mapping):

        sample_idx = sample_mapping[i]
        sequence_ids = batch.sequence_ids(i)

        # Default: no answer / answer not contained in this window
        start_token = 0
        end_token = 0

        if samples["answerable"].iloc[sample_idx]:

            answer_start = int(samples["answer_start"].iloc[sample_idx])

            answer_text = str(samples["answer"].iloc[sample_idx])

            answer_end = answer_start + len(answer_text)

            # Find first context token
            context_start = 0

            while sequence_ids[context_start] != 1:
                context_start += 1

            # Find last context token
            context_end = len(offsets) - 1

            while sequence_ids[context_end] != 1:
                context_end -= 1

            # Check whether the answer is inside this window
            if (
                answer_start >= offsets[context_start][0]
                and answer_end <= offsets[context_end][1]
            ):

                start_token = context_start

                while (
                    start_token < len(offsets)
                    and offsets[start_token][0] <= answer_start
                ):
                    start_token += 1

                start_token -= 1

                end_token = context_end

                while end_token >= 0 and offsets[end_token][1] >= answer_end:
                    end_token -= 1

                end_token += 1

        batch["start_positions"].append(start_token)

        batch["end_positions"].append(end_token)

        batch["example_index"].append(sample_idx)

        batch["offset_mapping"].append(offsets)

        batch["sequence_ids"].append(sequence_ids)

    features = []

    for i in range(len(batch["input_ids"])):

        features.append(
            {
                "input_ids": batch["input_ids"][i],
                "attention_mask": batch["attention_mask"][i],
                "start_positions": batch["start_positions"][i],
                "end_positions": batch["end_positions"][i],
                "example_index": batch["example_index"][i],
                "offset_mapping": batch["offset_mapping"][i],
                "sequence_ids": batch["sequence_ids"][i],
            }
        )

    return features


def train_collate_fn(inputs):
    input_ids = torch.tensor([x["input_ids"] for x in inputs])

    attention_mask = torch.tensor([x["attention_mask"] for x in inputs])

    start_positions = torch.tensor([x["start_positions"] for x in inputs])

    end_positions = torch.tensor([x["end_positions"] for x in inputs])

    # Trim padding to longest real sequence in this batch
    max_len = int(attention_mask.sum(dim=1).max())

    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": start_positions,
        "end_positions": end_positions,
    }


def val_collate_fn(inputs):

    input_ids = torch.tensor([x["input_ids"] for x in inputs])

    attention_mask = torch.tensor([x["attention_mask"] for x in inputs])

    max_len = int(attention_mask.sum(-1).max())

    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


class QATrainDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_squad(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)

    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
