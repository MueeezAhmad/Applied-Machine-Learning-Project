from transformers import pipeline


def extract_entities(text):

    ner_pipeline = pipeline(
        "ner",
        grouped_entities=True,
        framework="pt"
    )

    entities = ner_pipeline(text)

    results = []

    for entity in entities:

        results.append(
            f"{entity['word']} → {entity['entity_group']}"
        )

    return results