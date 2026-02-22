"""Audio file validation and preprocessing utilities."""

import os
import tempfile
import uuid
from typing import Tuple, Optional

from fastapi import UploadFile, HTTPException

from app.config import settings

# Allowed audio MIME types and extensions
ALLOWED_TYPES = {
    "audio/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "application/octet-stream": None,  # Accept but validate extension
}

ALLOWED_EXTENSIONS = {".webm", ".wav", ".mp3", ".ogg"}


async def validate_and_save_audio(file: UploadFile) -> Tuple[str, str]:
    """Validate uploaded audio file and save to a temporary location.

    Returns:
        Tuple of (temp_file_path, original_filename)

    Raises:
        HTTPException 400 if invalid format
        HTTPException 413 if file too large
    """
    # Check filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check extension
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format '{ext}'. Accepted formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file content
    content = await file.read()

    # Check file is not empty
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Check file size
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB"
        )

    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(temp_dir, temp_filename)

    with open(temp_path, "wb") as f:
        f.write(content)

    return temp_path, file.filename


def cleanup_temp_file(path: str) -> None:
    """Remove temporary audio file after processing."""
    try:
        if os.path.exists(path):
            os.remove(path)
            # Also remove parent temp dir if empty
            parent = os.path.dirname(path)
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
    except OSError:
        pass  # Best effort cleanup


def convert_to_wav_if_needed(audio_path: str) -> Tuple[str, Optional[str]]:
    """Convert WebM/MP3/OGG to WAV for librosa if needed.

    Returns:
        Tuple of (path_to_use_for_inference, path_to_cleanup_or_None)
        If the file is already .wav, returns (audio_path, None).
        If converted, returns (wav_path, wav_path) for cleanup.
    """
    _, ext = os.path.splitext(audio_path.lower())
    if ext == ".wav":
        return audio_path, None

    try:
        from pydub import AudioSegment
        import tempfile

        if ext == ".webm":
            audio = AudioSegment.from_file(audio_path, format="webm")
        elif ext == ".mp3":
            audio = AudioSegment.from_file(audio_path, format="mp3")
        elif ext == ".ogg":
            audio = AudioSegment.from_file(audio_path, format="ogg")
        else:
            return audio_path, None

        wav_path = audio_path.rsplit(".", 1)[0] + "_conv.wav"
        audio.export(wav_path, format="wav")
        return wav_path, wav_path
    except Exception:
        # Fallback: return original path; librosa may succeed via audioread
        return audio_path, None


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using librosa."""
    try:
        import librosa
        return float(librosa.get_duration(path=audio_path))
    except Exception:
        return 0.0
