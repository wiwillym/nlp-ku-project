import re

# (i): Majority-class baseline

def majority_class_baseline(df_train, df_eval):
    majority_class = df_train["answerable"].mean() >= 0.5
    predictions = [majority_class] * len(df_eval)
    return predictions


# (ii): Deterministic rule-based classifier

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
             "do", "does", "did", "in", "on", "at", "to", "of", "for", "and", "or",
             "what", "who", "when", "where", "why", "how", "which"}

def rule_based_classifier(question, context, lang, translator, threshold=0.3):
    translated_question = translator.translate(question, source_lang=lang)

    question_words = set(re.findall(r"[\w']+", translated_question.lower())) - STOPWORDS
    context_words = set(re.findall(r"[\w']+", context.lower()))

    matched_words = question_words & context_words
    overlap = len(matched_words) / len(question_words) if question_words else 0

    return overlap >= threshold