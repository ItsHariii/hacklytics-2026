"""Supabase service wrapper for storage, database, and realtime operations.

Currently uses placeholder configuration. Will be connected to a real
Supabase project once credentials are set up.
"""

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Supabase client — initialized lazily to avoid import errors when
# running without real credentials (e.g., during testing)
_supabase_client = None


def get_supabase_client():
    """Get or create the Supabase client instance."""
    global _supabase_client
    if _supabase_client is None:
        try:
            from supabase import create_client
            _supabase_client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY,
            )
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Supabase client: {e}")
            logger.warning("Running in offline mode — Supabase features disabled")
            return None
    return _supabase_client


def get_signed_audio_url(storage_path: str, expires_in: int = 3600) -> Optional[str]:
    """Generate a signed URL for a file in Supabase Storage.

    Returns the signed URL string, or None if Supabase is not configured or the request fails.
    """
    client = get_supabase_client()
    if client is None:
        return None

    try:
        response = client.storage.from_(settings.STORAGE_BUCKET).create_signed_url(
            storage_path, expires_in
        )
        if response and isinstance(response, dict) and "signedUrl" in response:
            return response["signedUrl"]
        # Newer API may return different structure
        if response and hasattr(response, "get"):
            return response.get("signedUrl") or response.get("signed_url")
        return None
    except Exception as e:
        logger.error(f"Failed to create signed URL for {storage_path}: {e}")
        return None


def is_supabase_configured() -> bool:
    """Return True if Supabase client is initialized (real credentials)."""
    return get_supabase_client() is not None


async def upload_audio(file_path: str, storage_path: str) -> Optional[str]:
    """Upload audio file to Supabase Storage.

    Returns the storage path on success, None on failure.
    """
    client = get_supabase_client()
    if client is None:
        logger.info(f"Mock upload: {storage_path}")
        return storage_path

    try:
        with open(file_path, "rb") as f:
            client.storage.from_(settings.STORAGE_BUCKET).upload(
                storage_path, f.read()
            )
        logger.info(f"Uploaded audio to: {storage_path}")
        return storage_path
    except Exception as e:
        logger.error(f"Storage upload failed: {e}")
        return None


async def store_recording(recording_data: dict) -> bool:
    """Store a recording record in the database."""
    client = get_supabase_client()
    if client is None:
        logger.info(f"Mock store recording: {recording_data.get('id')}")
        return True

    try:
        client.table("recordings").insert(recording_data).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to store recording: {e}")
        return False


async def store_classification(classification_data: dict) -> bool:
    """Store a classification result in the database."""
    client = get_supabase_client()
    if client is None:
        logger.info(f"Mock store classification: {classification_data.get('id')}")
        return True

    try:
        client.table("classifications").insert(classification_data).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to store classification: {e}")
        return False


async def broadcast_result(recording_id: str, result_data: dict) -> bool:
    """Broadcast classification result via Supabase Realtime.

    Channel: 'recording-results'
    Event: 'classification_complete'
    """
    client = get_supabase_client()
    if client is None:
        logger.info(f"Mock broadcast for recording: {recording_id}")
        return True

    try:
        channel = client.channel("recording-results")
        channel.send_broadcast(
            "classification_complete",
            {
                "recording_id": recording_id,
                "result": result_data,
            },
        )
        logger.info(f"Broadcast result for recording: {recording_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to broadcast result: {e}")
        return False
