from src.preprocessing import tokenise


def create_bio_labels(
    answerable,
    context,
    context_tokens,
    context_offsets,
    answer_start,
    answer,
):
    labels = ["O"] * len(context_tokens)

    if not answerable:
        return labels, "unanswerable"

    answer_start = int(answer_start)
    answer_end = answer_start + len(answer)

    covered = [
        i
        for i, (start, end) in enumerate(context_offsets)
        if start < answer_end and end > answer_start
    ]

    exact_alignment = (
        context_offsets[covered[0]][0] == answer_start
        and context_offsets[covered[-1]][1] == answer_end
    )

    # BIO labels
    labels[covered[0]] = "B-ANS"

    for i in covered[1:]:
        labels[i] = "I-ANS"

    alignment = "exact" if exact_alignment else "partial_token"

    return labels, alignment


def test_bio_conversion():

    # 1. Answer at character 0
    context = "Paris is in France."
    tokens, offsets = tokenise(context, "en")

    labels, _ = create_bio_labels(
        True,
        context,
        tokens,
        offsets,
        0,
        "Paris",
    )

    assert labels[0] == "B-ANS"

    # 2. Multi-token answer
    context = "I live in New York."
    tokens, offsets = tokenise(context, "en")

    answer = "New York"
    start = context.index(answer)

    labels, _ = create_bio_labels(
        True,
        context,
        tokens,
        offsets,
        start,
        answer,
    )

    assert labels[tokens.index("New")] == "B-ANS"
    assert labels[tokens.index("York")] == "I-ANS"

    # 3. Punctuation adjacent to answer
    context = "I visited Paris, France."
    tokens, offsets = tokenise(context, "en")

    answer = "Paris"
    start = context.index(answer)

    labels, _ = create_bio_labels(
        True,
        context,
        tokens,
        offsets,
        start,
        answer,
    )

    paris_idx = tokens.index("Paris")
    comma_idx = tokens.index(",")

    assert labels[paris_idx] == "B-ANS"
    assert labels[comma_idx] == "O"

    # 4. Unanswerable
    context = "Paris is in France."
    tokens, offsets = tokenise(context, "en")

    labels, _ = create_bio_labels(
        False,
        context,
        tokens,
        offsets,
        -1,
        "",
    )

    assert all(label == "O" for label in labels)

    print("All BIO conversion tests passed.")
