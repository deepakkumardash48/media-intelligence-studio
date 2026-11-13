import os
import sys

# Ensure UTF-8 output encoding for Windows terminals with special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Configure ffmpeg path if installed via winget or default directories
FFMPEG_PATH = r"C:\Users\Deepak\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
if os.path.exists(FFMPEG_PATH) and FFMPEG_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

import yt_dlp
from pydub import AudioSegment

if os.path.exists(FFMPEG_PATH):
    AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(
        DOWNLOAD_DIR, "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "ffmpeg_location": FFMPEG_PATH if os.path.exists(FFMPEG_PATH) else None,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(
            ".webm", ".wav"
        )
        return filename


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
    
    # If audio is already shorter than chunk length, return as single chunk
    if len(audio) <= chunk_lenms:
        return [wav_path]

    base_name = os.path.splitext(wav_path)[0]
    chunks = []
    
    # Step through audio by chunk_lenms (milliseconds)
    for i, start in enumerate(range(0, len(audio), chunk_lenms)):
        chunk = audio[start:start + chunk_lenms]
        chunk_path = f"{base_name}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        
    return chunks


# Takes input from the user (either YouTube URL or local file) and returns chunks
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
