from flask import Flask, render_template, request, jsonify
import os, json

from utils.transcription import transcribe_audio
from utils.summarization import summarize_text
from utils.intent import detect_intent
from utils.sentiment import analyze_sentiment
from utils.diarization import separate_speakers

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 1️⃣ Transcribe audio
    full_text = transcribe_audio(file_path)

    # 2️⃣ Speaker diarization
    speaker_segments = separate_speakers(file_path)

    # 3️⃣ Summarization
    combined_summary = summarize_text(full_text)

    # 4️⃣ Intent detection
    intents = detect_intent(combined_summary)

    # 5️⃣ Sentiment analysis
    sentiment = analyze_sentiment(combined_summary)

    result = {
        "summary": combined_summary,
        "intent": intents,
        "sentiment": sentiment,
        "speakers": speaker_segments
    }

    # Save JSON
    os.makedirs("data", exist_ok=True)
    with open("data/transcription_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
