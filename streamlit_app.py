import streamlit as st
from faster_whisper import WhisperModel
from transformers import pipeline
from textblob import TextBlob
from langdetect import detect

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

whisper_model = load_whisper()
summarizer = load_summarizer()

# ----------------------------
# Helper functions
# ----------------------------
def transcribe(file):
    segments, info = whisper_model.transcribe(file)
    text = " ".join([seg.text for seg in segments])
    return text

def analyze_sentiment(text):
    blob = TextBlob(text)
    p = blob.sentiment.polarity
    if p > 0.2: return "Positive"
    if p < -0.2: return "Negative"
    return "Neutral"

def detect_intent(text):
    text = text.lower()
    if any(k in text for k in ["exchange", "refund", "return"]): return "Exchange request"
    if any(k in text for k in ["order", "delivery"]): return "Order status"
    if any(k in text for k in ["price", "cost"]): return "Product inquiry"
    return "General conversation"

# ----------------------------
# UI
# ----------------------------
st.title("🎙 Voice Analysis Dashboard")

file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])

if file:
    st.audio(file)

    st.write("Transcribing...")
    text = transcribe(file)

    st.subheader("Transcription")
    st.write(text)

    st.subheader("Language")
    st.write(detect(text))

    st.subheader("Sentiment")
    st.write(analyze_sentiment(text))

    st.subheader("Intent")
    st.write(detect_intent(text))

    st.subheader("Summary")
    summary = summarizer(text[:1024])[0]['summary_text']
    st.write(summary)
