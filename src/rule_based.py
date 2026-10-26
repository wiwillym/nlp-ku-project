import re

# (i): Majority-class baseline


def majority_class_baseline(df_train, df_eval):
    majority_class = df_train["answerable"].mean() >= 0.5
    predictions = [majority_class] * len(df_eval)
    return predictions


# (ii): Deterministic rule-based classifier

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "in",
    "on",
    "at",
    "to",
    "of",
    "for",
    "and",
    "or",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
}


def classify_translated_question(translated_question, context, threshold=0.3):
    question_words = set(re.findall(r"[\w']+", translated_question.lower())) - STOPWORDS
    context_words = set(re.findall(r"[\w']+", context.lower()))

    matched_words = question_words & context_words
    overlap = len(matched_words) / len(question_words) if question_words else 0

    return overlap >= threshold


def rule_based_classifier(question, context, lang, translator, threshold=0.3):
    translated_question = translator.translate(question, source_lang=lang)
    return classify_translated_question(translated_question, context, threshold)


def rule_based_classifier_batch(df, translator, threshold=0.3, batch_size=8):
    predictions = [False] * len(df)

    for lang, positions in df.groupby("lang", sort=False).indices.items():

        questions = df.iloc[positions]["question"].tolist()
        contexts = df.iloc[positions]["context"].tolist()

        translated_questions = translator.translate_batch(
            questions,
            source_lang=lang,
            batch_size=batch_size,
        )

        for pos, translated_question, context in zip(
            positions, translated_questions, contexts
        ):
            predictions[pos] = classify_translated_question(
                translated_question,
                context,
                threshold,
            )

    return predictions
