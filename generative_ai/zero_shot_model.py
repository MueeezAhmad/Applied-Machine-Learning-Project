from transformers import pipeline


classifier = pipeline(
    "zero-shot-classification"
)


def classify_text(text):

    labels = [

        "sports",

        "technology",

        "politics",

        "education"

    ]

    result = classifier(
        text,
        labels
    )

    return result["labels"][0]