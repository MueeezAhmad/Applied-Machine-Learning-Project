from transformers import pipeline


sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)


def analyze_sentiment(text):

    result = sentiment(text)

    return result[0]["label"]