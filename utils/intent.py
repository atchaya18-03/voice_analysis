# streamlit_app.py
import streamlit as st
import os, json, time
from utils.transcription import transcribe_audio
from utils.diarization import separate_speakers
from utils.summarization import summarize_text
from utils.sentiment import analyze_sentiment
from utils.intent import detect_intent

st.set_page_config(page_title="Voice Analysis", layout="wide")

st.title("Voice Analysis — Agent / Customer Dashboard")

uploaded = st.file_uploader("Upload audio file (wav/mp3)", type=["wav","mp3","m4a","flac"])
if uploaded is None:
    st.info("Upload an audio file to start the pipeline.")
    st.stop()

# Save upload
os.makedirs("uploads", exist_ok=True)
file_path = os.path.join("uploads", uploaded.name)
with open(file_path, "wb") as f:
    f.write(uploaded.getbuffer())

st.sidebar.header("Pipeline Steps")
steps = [
    "Transcription",
    "Language detection",
    "Diarization (speaker split)",
    "Speaker -> Agent/Customer mapping",
    "Per-speaker analysis (sentiment/intent)",
    "Summarization",
    "Resolution detection",
]
status = {s: "pending" for s in steps}

# Run pipeline
with st.spinner("Running transcription..."):
    segments, full_text, lang = transcribe_audio(file_path)
    status["Transcription"] = "done"
    time.sleep(0.2)
st.success("Transcription complete")

st.sidebar.write(status)
st.subheader("Step 1 — Transcription (timestamped segments)")
st.write(f"Detected language (raw): **{lang}**")
st.code("\n".join([f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments]), language="text")

status["Language detection"] = "done"

with st.spinner("Running speaker diarization..."):
    diarization = separate_speakers(file_path)
    status["Diarization (speaker split)"] = "done"
st.success("Diarization complete")

# show diarization timeline in simple table
st.subheader("Step 2 — Speaker segments")
for speaker, segs in diarization.items():
    st.write(f"**{speaker}**")
    for seg in segs:
        st.write(f"- {seg['start']:.2f}s → {seg['end']:.2f}s")

# Map speakers to AGENT / CUSTOMER heuristically
st.subheader("Step 3 — Map speakers to Agent / Customer (heuristic)")
# Heuristic: first speaker -> AGENT, else CUSTOMER. If only one speaker, label AGENT.
speaker_labels = list(diarization.keys())
mapping = {}
if len(speaker_labels) == 0:
    mapping = {"unknown":"unknown"}
elif len(speaker_labels) == 1:
    mapping[speaker_labels[0]] = "AGENT"
else:
    mapping[speaker_labels[0]] = "AGENT"
    for s in speaker_labels[1:]:
        mapping[s] = "CUSTOMER"

st.write(mapping)
status["Speaker -> Agent/Customer mapping"] = "done"

# Aggregate text per speaker using timestamps: naive approach
st.subheader("Step 4 — Build per-speaker transcript (approx.)")
per_speaker_text = {}
for s_label, segs in diarization.items():
    texts = []
    for seg in segs:
        # find overlapping transcript segments and join text (simple)
        for t in segments:
            if (t["start"] < seg["end"] and t["end"] > seg["start"]):
                texts.append(t["text"])
    per_speaker_text[s_label] = " ".join(texts).strip()
    st.write(f"**{s_label} ({mapping.get(s_label,'unknown')})**")
    st.write(per_speaker_text[s_label] or "(no text found for segments)")

# Per-speaker analysis
st.subheader("Step 5 — Per-speaker analysis")
per_speaker_analysis = {}
for s_label, text in per_speaker_text.items():
    sent = analyze_sentiment(text)
    intents = detect_intent(text)
    per_speaker_analysis[s_label] = {"text": text, "sentiment": sent, "intents": intents, "role": mapping.get(s_label,"unknown")}
    st.write(f"**{s_label} ({mapping.get(s_label,'unknown')})**")
    st.write(f"- Sentiment: {sent}")
    st.write(f"- Intents: {intents}")

status["Per-speaker analysis (sentiment/intent)"] = "done"

with st.spinner("Generating summary..."):
    summary = summarize_text(full_text)
    status["Summarization"] = "done"
st.success("Summary ready")
st.subheader("Step 6 — Summary")
st.write(summary)

# Very simple resolution detection
st.subheader("Step 7 — Resolution detection & Feedback")
lower = full_text.lower()
resolved = any(kw in lower for kw in ["resolved", "fixed", "thank you", "thanks", "issue closed", "closed"])
status["Resolution detection"] = "done"
st.write("Purpose resolved?" , resolved)

# Aggregate final JSON
final = {
    "file": uploaded.name,
    "language": lang,
    "segments": segments,
    "diarization": diarization,
    "speaker_mapping": mapping,
    "per_speaker": per_speaker_analysis,
    "summary": summary,
    "resolved": resolved,
    "pipeline_status": status
}

st.download_button("Download result JSON", json.dumps(final, indent=2, ensure_ascii=False), file_name="analysis_result.json")

st.sidebar.write("Pipeline status")
st.sidebar.json(status)
