# utils/diarization.py
import os
from typing import Dict, List, Tuple

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

def try_pyannote_pipeline(file_path):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
    diarization = pipeline(file_path)
    # produce list of (start,end,label)
    items = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        items.append((float(turn.start), float(turn.end), str(speaker)))
    return items

def fallback_speechbrain_diarization(file_path, chunk_size_s: int = 3, n_clusters: int = 2):
    """
    Lightweight fallback: split into short chunks, compute speaker embeddings with speechbrain
    and cluster them with KMeans. Returns list of (start,end,label).
    """
    import torchaudio
    import numpy as np
    from speechbrain.pretrained import EncoderClassifier
    from sklearn.cluster import KMeans

    waveform, sr = torchaudio.load(file_path)
    duration = waveform.shape[1] / sr
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_speechbrain")
    chunks = []
    embeddings = []
    starts = []
    for start in range(0, int(duration), chunk_size_s):
        end = min(start + chunk_size_s, int(duration))
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        chunk = waveform[:, start_idx:end_idx]
        if chunk.shape[1] < 1600:  # too short
            continue
        with np.errstate(all='ignore'):
            emb = classifier.encode_batch(chunk).squeeze().detach().cpu().numpy()
        embeddings.append(emb)
        starts.append((start, end))
    if not embeddings:
        return []
    embeddings = np.stack(embeddings)
    k = min(n_clusters, embeddings.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    items = []
    for (s,e),lab in zip(starts, labels):
        items.append((float(s), float(e), f"speaker_{lab}"))
    return items

def separate_speakers(file_path) -> Dict[str, List[Tuple[float,float,str]]]:
    """
    Returns a dict mapping speaker_label -> list of (start,end,label)
    """
    items = []
    if HF_TOKEN:
        try:
            items = try_pyannote_pipeline(file_path)
        except Exception as e:
            # fallback
            items = fallback_speechbrain_diarization(file_path)
    else:
        items = fallback_speechbrain_diarization(file_path)

    # Convert to dict
    out = {}
    for start, end, label in items:
        out.setdefault(label, []).append({"start": start, "end": end})
    return out
