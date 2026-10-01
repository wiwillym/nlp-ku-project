"""
lm_utils.py — shared vocabulary/UNK strategy + n-gram language models
for the Week 38 Language Modelling task.

SHARED STRATEGY:
  - Vocabulary is built ONLY from the training split's question tokens,
    for the ONE chosen language (e.g. Arabic).
  - Any train token with count < MIN_COUNT is folded into <UNK> when
    building the vocab (i.e. it never becomes its own vocab entry).
  - The SAME vocab is then used to UNK-map both train and validation
    token lists before fitting/evaluating any model.
  - Every group member's model must be trained/evaluated on tokens
    produced by build_vocab() + apply_unk() below, so perplexities
    are comparable across configs.

Usage sketch:

    import pandas as pd
    from lm_utils import build_vocab, apply_unk, add_boundaries, UnigramLM

    ar_train = df_train_tokenised.loc[df_train_tokenised["lang"] == "ar", "question_tokens"].tolist()
    ar_val   = df_val_tokenised.loc[df_val_tokenised["lang"] == "ar", "question_tokens"].tolist()

    vocab = build_vocab(ar_train, min_count=2)

    train_unk = [apply_unk(toks, vocab) for toks in ar_train]
    val_unk   = [apply_unk(toks, vocab) for toks in ar_val]

    model = UnigramLM(k=<value>)
    model.fit(train_unk, vocab)
    ppl = model.perplexity(val_unk)
    print("Unigram val perplexity:", ppl)
"""

from collections import Counter
import math

UNK = "<UNK>"
BOS = "<s>"
EOS = "</s>"
MIN_COUNT = 2


def build_vocab(token_lists, min_count=MIN_COUNT):
    """
    Build the shared vocabulary from TRAINING question tokens only.

    token_lists: list[list[str]] — tokenised training questions for one language.
    Returns: set[str] — vocab, always includes UNK/BOS/EOS.
    """
    counts = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    vocab = {tok for tok, c in counts.items() if c >= min_count}
    vocab.update({UNK, BOS, EOS})
    return vocab


def apply_unk(tokens, vocab):
    """
    Map every token not in the shared vocab to <UNK>.
    Apply this with the SAME vocab to both train and validation tokens.
    """
    return [tok if tok in vocab else UNK for tok in tokens]


def add_boundaries(tokens, order=1):
    """
    Add sentence boundary markers so n-gram models have context at the
    start/end of a question.

    order=1 (unigram): boundaries aren't required, but adding one BOS/EOS
                        is harmless and keeps things uniform.
    order=2 (bigram):   1 BOS + 1 EOS.
    order=3 (trigram):  2 BOS + 1 EOS, so the first real token still has
                        a full 2-word history.
    """
    n_bos = max(order - 1, 1)
    return [BOS] * n_bos + list(tokens) + [EOS]


class UnigramLM:
    """
    Unigram language model with add-k smoothing.
    This ignores all context. it's the REQUIRED baseline for the group.
    """

    def __init__(self, k: float = 1.0):
        self.k = k
        self.counts = Counter()
        self.total = 0
        self.vocab = None

    def fit(self, token_lists, vocab):
        """
        token_lists: list[list[str]] — already UNK-mapped training questions.
        vocab: the shared vocab set from build_vocab().
        """
        self.vocab = vocab
        self.counts = Counter()
        for tokens in token_lists:
            self.counts.update(tokens)
        self.total = sum(self.counts.values())

    def prob(self, word: str) -> float:
        V = len(self.vocab)
        return (self.counts[word] + self.k) / (self.total + self.k * V)

    def sentence_logprob(self, tokens) -> float:
        return sum(math.log(self.prob(tok)) for tok in tokens)

    def perplexity(self, token_lists) -> float:
        """
        Per-token perplexity over a set of already UNK-mapped sentences.
        token_lists: list[list[str]] (e.g. validation questions).
        """
        total_logprob = 0.0
        total_tokens = 0
        for tokens in token_lists:
            total_logprob += self.sentence_logprob(tokens)
            total_tokens += len(tokens)
        avg_neg_logprob = -total_logprob / total_tokens
        return math.exp(avg_neg_logprob)


class BigramLM:

    def __init__(self, k=1.0):
        self.k = k
        self.bigram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = None

    def fit(self, token_lists, vocab):
        self.vocab = vocab
        self.bigram_counts = Counter()
        self.context_counts = Counter()

        for tokens in token_lists:

            sequence = [BOS] + list(tokens)

            for prev, word in zip(sequence[:-1], sequence[1:]):
                self.bigram_counts[(prev, word)] += 1
                self.context_counts[prev] += 1

    def prob(self, word, prev):
        V = len(self.vocab)

        numerator = self.bigram_counts[(prev, word)] + self.k

        denominator = self.context_counts[prev] + self.k * V

        return numerator / denominator

    def sentence_logprob(self, tokens):
        sequence = [BOS] + list(tokens)

        logprob = 0.0

        for prev, word in zip(sequence[:-1], sequence[1:]):
            logprob += math.log(self.prob(word, prev))

        return logprob

    def perplexity(self, token_lists):
        total_logprob = 0.0
        total_tokens = 0

        for tokens in token_lists:
            total_logprob += self.sentence_logprob(tokens)

            total_tokens += len(tokens)

        avg_neg_logprob = -total_logprob / total_tokens

        return math.exp(avg_neg_logprob)

class TrigramLM:

    def __init__(self, k=1.0):
            self.k = k
            self.trigram_counts = Counter()
            self.context_counts = Counter()
            self.vocab = None
    
    def fit(self, token_lists, vocab):
            self.vocab = vocab
            self.trigram_counts = Counter()
            self.context_counts = Counter()
    
            for tokens in token_lists:
    
                sequence = [BOS, BOS] + list(tokens)
    
                for prev_prev, prev, word in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
                    self.trigram_counts[(prev_prev, prev, word)] += 1
                    self.context_counts[(prev_prev, prev)] += 1
    
    def prob(self, word, prev_prev, prev):
            V = len(self.vocab)
    
            numerator = self.trigram_counts[(prev_prev, prev, word)] + self.k
    
            denominator = self.context_counts[(prev_prev, prev)] + self.k * V
    
            return numerator / denominator
    
    def sentence_logprob(self, tokens):
            sequence = [BOS, BOS] + list(tokens)
    
            logprob = 0.0
    
            for prev_prev, prev, word in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
                logprob += math.log(self.prob(word, prev_prev, prev))
    
            return logprob
    
    def perplexity(self, token_lists):
           total_logprob = 0.0
           total_tokens = 0
   
           for tokens in token_lists:
               total_logprob += self.sentence_logprob(tokens)
   
               total_tokens += len(tokens)
   
           avg_neg_logprob = -total_logprob / total_tokens
   
           return math.exp(avg_neg_logprob)