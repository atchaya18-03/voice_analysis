# utils/summarization.py
from transformers import pipeline

# lightweight summarizer
_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text, max_input_chars=4000):
    if not text:
        return ""
    # Shorten if too long
    text = text[:max_input_chars]
    result = _summarizer(text, max_length=150, min_length=30, do_sample=False)
    return result[0]["summary_text"]
