import os
import requests
import torch
import whisper
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", os.getenv("SARVAM_STT_MODLE", "saaras:v2.5"))

_model = None


def load_model():
    """Load and cache the Whisper model."""
    global _model  

    if _model is None: 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{WHISPER_MODEL}' on {device.upper()}...")
        _model = whisper.load_model(WHISPER_MODEL, device=device) 
        print("Whisper model loaded successfully.")
    return _model 


def detect_audio_language(audio_path: str) -> str:
    """Detect spoken language in audio using Whisper."""
    model = load_model()
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    return detected_lang


def transcribe_chunk_whisper(chunk_path: str, task: str = "transcribe") -> str:
    """Transcribe or translate an audio chunk using local Whisper."""
    model = load_model()  
    use_fp16 = torch.cuda.is_available()
    result = model.transcribe(chunk_path, task=task, fp16=use_fp16)  
    return result.get("text", "").strip()  


def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts <=30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk according to user preference and audio language:
    - English: Whisper runs first. If Whisper encounters an error, falls back to Sarvam AI.
    - Hinglish / Indian languages: Sarvam AI runs first. If Sarvam fails, falls back to Whisper translate.
    - Auto: Detects language; if English ('en'), uses Whisper first; otherwise Sarvam first.
    """
    target_lang = language.lower().strip()

    if target_lang == "auto":
        try:
            detected = detect_audio_language(chunk_path)
            print(f"  [Auto-detect] Language detected: '{detected}'")
            target_lang = "english" if detected == "en" else "hinglish"
        except Exception as e:
            print(f"  [Auto-detect error: {e}] Defaulting to English (Whisper first)")
            target_lang = "english"

    # Route based on language
    if target_lang in ("hinglish", "hindi"):
        try:
            print("  → Transcribing with Sarvam AI...")
            return transcribe_chunk_sarvam(chunk_path)
        except Exception as e:
            print(f"  ⚠️ Sarvam failed: {e}. Falling back to Whisper (task='translate')...")
            return transcribe_chunk_whisper(chunk_path, task="translate")
    else:
        # English: Whisper works first, Sarvam is fallback
        try:
            print("  → Transcribing with Whisper (English)...")
            return transcribe_chunk_whisper(chunk_path, task="transcribe")
        except Exception as e:
            print(f"  ⚠️ Whisper failed: {e}. Falling back to Sarvam AI...")
            return transcribe_chunk_sarvam(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Transcribe all chunks with routing and fallback."""
    full_transcript = "" 

    mode_info = f"language mode '{language}' (English -> Whisper first, Hinglish -> Sarvam first, with automatic fallback)"
    print(f"\n🎙️ Starting transcription using {mode_info}...")

    for i, chunk in enumerate(chunks):  
        print(f"\n[Chunk {i + 1}/{len(chunks)}]: {os.path.basename(chunk)}")
        text = transcribe_chunk(chunk, language=language)  
        full_transcript += text + " "  

    print("\n✅ Transcription complete.")
    return full_transcript.strip()
