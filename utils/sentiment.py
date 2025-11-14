# utils/sentiment.py
from textblob import TextBlob

def analyze_sentiment(text):
    """
    Returns dict: {"sentiment":"positive/neutral/negative","polarity":float}
    """
    if not text:
        return {"sentiment": "neutral", "polarity": 0.0}
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {"sentiment": sentiment, "polarity": polarity}
