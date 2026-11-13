import os
import sys
import tempfile
import streamlit as st
import yt_dlp
from pydub import AudioSegment

# Ensure UTF-8 output encoding for Windows terminals with special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Configure ffmpeg path if installed locally on Windows
FFMPEG_PATH = r"C:\Users\Deepak\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
if os.path.exists(FFMPEG_PATH) and FFMPEG_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

if os.path.exists(FFMPEG_PATH):
    AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_cookies_file() -> str | None:
    """Checks for a local cookies.txt or Streamlit Secret YOUTUBE_COOKIES."""
    if os.path.exists("cookies.txt"):
        return "cookies.txt"
    try:
        if "YOUTUBE_COOKIES" in st.secrets and st.secrets["YOUTUBE_COOKIES"].strip():
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
            tmp.write(st.secrets["YOUTUBE_COOKIES"])
            tmp.close()
            return tmp.name
    except Exception:
        pass
    return None


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    cookie_path = get_cookies_file()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Bypass YouTube 403 Forbidden on cloud servers using mobile/TV player clients
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "tv", "web"],
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    if os.path.exists(FFMPEG_PATH):
        ydl_opts["ffmpeg_location"] = FFMPEG_PATH

    if cookie_path:
        ydl_opts["cookiefile"] = cookie_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_filename = ydl.prepare_filename(info)
        wav_filename = os.path.splitext(raw_filename)[0] + ".wav"
        return wav_filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split a WAV file into smaller chunks of specified minutes."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_lenms = int(chunk_minutes * 60 * 1000)

    if len(audio) <= chunk_lenms:
        return [wav_path]

    base_name = os.path.splitext(wav_path)[0]
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_lenms)):
        chunk = audio[start : start + chunk_lenms]
        chunk_path = f"{base_name}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("detected youtube url, downloading audio..")
        wav_path = download_youtube_audio(source)
    else:
        print("detected local file, converting to wav....")
        wav_path = convert_to_wav(source)

    print("chunking audio into 10 minutes chunks..")
    chunks = chunk_audio(wav_path)
    print(f"audio ready - {len(chunks)} chunks created")
    return chunks