import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score


class PretrainedGRUClassifier(nn.Module):
    def __init__(
        self,
        pretrained_embeddings,
        hidden_dim=64,
        dropout_prob=0.3,
        padding_idx=1,
    ):
        super().__init__()

        embedding_dim = pretrained_embeddings.shape[1]

        self.embeddings = nn.Embedding.from_pretrained(
            pretrained_embeddings,
            padding_idx=padding_idx,
            freeze=True,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(4 * hidden_dim, 1)

    def encode(self, input_ids, lengths):
        embeddings = self.embeddings(input_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)

        return hidden[-1]

    def forward(
        self,
        question_ids,
        question_lengths,
        context_ids,
        context_lengths,
    ):
        q_repr = self.encode(question_ids, question_lengths)

        c_repr = self.encode(context_ids, context_lengths)

        combined = torch.cat(
            [
                q_repr,
                c_repr,
                torch.abs(q_repr - c_repr),
                q_repr * c_repr,
            ],
            dim=-1,
        )

        combined = self.dropout(combined)

        logits = self.classifier(combined).squeeze(-1)

        return logits


def train_one_epoch_answerability(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        logits = model(
            batch["question_ids"].to(device),
            batch["question_lengths"].to(device),
            batch["context_ids"].to(device),
            batch["context_lengths"].to(device),
        )

        labels = batch["labels"].to(device)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def predict_answerability(model, dataloader, device, threshold=0.5):
    model.eval()

    predictions = []
    probabilities = []
    gold = []
    languages = []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                batch["question_ids"].to(device),
                batch["question_lengths"].to(device),
                batch["context_ids"].to(device),
                batch["context_lengths"].to(device),
            )

            probs = torch.sigmoid(logits)
            pred = (probs >= threshold).long()

            predictions.extend(pred.cpu().numpy())

            probabilities.extend(probs.cpu().numpy())

            gold.extend(batch["labels"].long().numpy())

            languages.extend(batch["lang"])

    return (
        np.asarray(predictions),
        np.asarray(probabilities),
        np.asarray(gold),
        np.asarray(languages),
    )


def evaluate_macro_f1(model, dataloader, device, threshold=0.5):
    pred, _, gold, _ = predict_answerability(model, dataloader, device, threshold=threshold)
    return f1_score(gold, pred, average="macro")
