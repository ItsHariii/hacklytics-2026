"""GET /api/recordings — List all; GET /api/recordings/{id} — Return full classification."""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from app.models.schemas import RecordingResponse, Recording, ClassificationResult
from app.services.recording_cache import get as cache_get, list_all as cache_list_all
from app.services import supabase_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/recordings", response_model=List[RecordingResponse])
async def list_recordings(
    limit: int = Query(default=50, ge=1, le=100, description="Max recordings to return"),
    patient_id: Optional[str] = Query(default=None, description="Filter by patient ID"),
):
    """List recent recordings from cache (real model results).

    Returns recordings from the analyze flow cache. Empty when no recordings yet.
    """
    items = cache_list_all()
    result: List[RecordingResponse] = []
    for item in items:
        if patient_id and item.get("recording", {}).get("patient_id") != patient_id:
            continue
        if len(result) >= limit:
            break
        result.append(
            RecordingResponse(
                recording=Recording(**item["recording"]),
                result=ClassificationResult(**item["result"]),
            )
        )
    return result


@router.get("/recordings/{recording_id}/audio")
async def get_recording_audio(recording_id: str):
    """Stream audio for a recording. Redirects to Supabase signed URL or serves from local storage."""
    cached = cache_get(recording_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Recording not found")

    # 1. Local file (when Supabase not configured)
    local_path = cached.get("local_audio_path")
    if local_path:
        path = Path(local_path)
        if path.exists():
            return FileResponse(
                path,
                media_type="audio/wav",
                filename=f"{recording_id}.wav",
            )
        logger.warning(f"Local audio file missing: {local_path}")

    # 2. Supabase signed URL
    rec = cached.get("recording", {})
    audio_path = rec.get("audio_path")
    if audio_path and supabase_service.is_supabase_configured():
        signed_url = supabase_service.get_signed_audio_url(audio_path)
        if signed_url:
            return RedirectResponse(url=signed_url, status_code=302)

    raise HTTPException(status_code=404, detail="Audio not available for this recording")


@router.get("/recordings/{recording_id}", response_model=RecordingResponse)
async def get_recording(recording_id: str):
    """Get the full classification result for a specific recording.

    Checks in-memory cache first (populated by analyze flow), then Supabase,
    then falls back to mock for unknown IDs.
    """
    # 1. Check in-memory cache (from recent analyze flow)
    cached = cache_get(recording_id)
    if cached:
        return RecordingResponse(
            recording=Recording(**cached["recording"]),
            result=ClassificationResult(**cached["result"]),
        )

    # 2. TODO: Query Supabase when configured

    # 3. Unknown recording_id — return 404 (frontend polls until cache populated)
    raise HTTPException(status_code=404, detail="Recording not found or still processing")
