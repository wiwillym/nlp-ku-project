import torch
import torch.nn as nn
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


def token_features_from_ids(
    question_ids,
    context_ids,
    i,
):
    token = context_ids[i]

    features = {
        "bias": 1.0,
        f"word_id={token}": 1.0,
        "in_question": float(token in question_ids),
        "relative_position": i / max(len(context_ids), 1),
    }

    if i > 0:
        features[f"prev_word_id={context_ids[i - 1]}"] = 1.0
    else:
        features["BOS"] = 1.0

    if i < len(context_ids) - 1:
        features[f"next_word_id={context_ids[i + 1]}"] = 1.0
    else:
        features["EOS"] = 1.0

    for q_id in set(question_ids):
        features[f"question_word_id={q_id}"] = 1.0

    return features


class EmptySpanBaseline(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, question_ids, context_ids):
        batch_size, context_length = context_ids.shape

        logits = torch.zeros(
            batch_size,
            context_length,
            self.num_labels,
            device=context_ids.device,
        )

        logits[:, :, 0] = 1.0

        return logits


class SequenceLabeller(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        context_dim=128,
        hidden_dim=128,
        num_labels=3,
        padding_idx=0,
    ):
        super().__init__()
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

        self.context_cnn = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=context_dim,
            kernel_size=3,
            padding=1,
        )

        self.classifier = nn.Sequential(
            nn.Linear(context_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        question_ids,
        context_ids,
    ):
        question_emb = self.embedding(question_ids)
        question_mask = (question_ids != self.padding_idx).unsqueeze(-1)
        question_sum = (question_emb * question_mask).sum(dim=1)
        question_lengths = question_mask.sum(dim=1).clamp(min=1)
        question_repr = question_sum / question_lengths

        context_emb = self.embedding(context_ids)
        context_repr = self.context_cnn(context_emb.transpose(1, 2))
        context_repr = torch.relu(context_repr)
        context_repr = context_repr.transpose(1, 2)

        question_repeated = question_repr.unsqueeze(1).expand(
            -1,
            context_repr.size(1),
            -1,
        )

        combined = torch.cat(
            [context_repr, question_repeated],
            dim=-1,
        )

        logits = self.classifier(combined)

        return logits


class LocalSequenceLabeller:
    def __init__(
        self,
        class_weight=None,
        C=1.0,
        max_iter=1000,
    ):
        self.vectorizer = DictVectorizer()
        self.label_encoder = LabelEncoder()

        self.lr = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
        )

    def _batch_to_features_and_labels(
        self,
        batch,
    ):
        question_ids = batch["question_ids"]
        context_ids = batch["context_ids"]
        labels = batch["labels"]

        X = []
        y = []

        for q_ids, c_ids, gold in zip(
            question_ids,
            context_ids,
            labels,
        ):
            # Remove padding
            q_ids = q_ids[q_ids != 0].tolist()

            valid = gold != -100

            c_ids = c_ids[valid].tolist()
            gold = gold[valid].tolist()

            for i in range(len(c_ids)):
                X.append(
                    token_features_from_ids(
                        q_ids,
                        c_ids,
                        i,
                    )
                )

                y.append(gold[i])

        return X, y

    def fit(self, dataloader):
        all_X = []
        all_y = []

        for batch in dataloader:
            X, y = self._batch_to_features_and_labels(batch)

            all_X.extend(X)
            all_y.extend(y)

        X_vectorized = self.vectorizer.fit_transform(all_X)

        y_encoded = self.label_encoder.fit_transform(all_y)

        self.lr.fit(
            X_vectorized,
            y_encoded,
        )

        return self

    def predict_batch(self, batch):
        question_ids = batch["question_ids"]
        context_ids = batch["context_ids"]
        labels = batch["labels"]

        batch_predictions = []

        for q_ids, c_ids, gold in zip(
            question_ids,
            context_ids,
            labels,
        ):
            q_ids = q_ids[q_ids != 0].tolist()

            valid = gold != -100
            c_ids = c_ids[valid].tolist()

            X = [
                token_features_from_ids(
                    q_ids,
                    c_ids,
                    i,
                )
                for i in range(len(c_ids))
            ]

            X_vectorized = self.vectorizer.transform(X)

            pred_encoded = self.lr.predict(X_vectorized)

            predictions = self.label_encoder.inverse_transform(pred_encoded)

            batch_predictions.append(
                torch.tensor(
                    predictions,
                    dtype=torch.long,
                )
            )

        return batch_predictions
