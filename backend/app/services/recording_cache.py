"""In-memory cache for recording results.

Used when Supabase is not configured — allows GET /api/recordings/{id}
to return real classification results from the analyze flow.
"""

from typing import Dict, Optional

# recording_id -> RecordingResponse (as dict for JSON-serializable storage)
_cache: Dict[str, dict] = {}


def put(
    recording_id: str,
    recording_data: dict,
    result_data: dict,
    local_audio_path: Optional[str] = None,
) -> None:
    """Store a recording + result for later retrieval.

    local_audio_path: Optional path to local WAV file (when Supabase not configured).
    """
    _cache[recording_id] = {
        "recording": recording_data,
        "result": result_data,
        "local_audio_path": local_audio_path,
    }


def get(recording_id: str) -> Optional[dict]:
    """Retrieve cached recording response, or None."""
    return _cache.get(recording_id)


def list_all() -> list:
    """Return all cached recordings, newest first.

    Each item is a dict with keys: recording, result (matching RecordingResponse shape).
    """
    items = list(_cache.values())
    # Sort by created_at descending (newest first)
    items.sort(
        key=lambda x: x.get("recording", {}).get("created_at", ""),
        reverse=True,
    )
    return items
