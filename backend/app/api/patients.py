"""GET /api/patients/{patient_id}/history — Return recording history + trend data."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from app.models.schemas import (
    PatientHistoryResponse,
    Patient,
    RecordingResponse,
    Recording,
    ClassificationResult,
    TrendPoint,
)
from app.services.recording_cache import list_all as cache_list_all

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_trend_from_recordings(recordings: list[RecordingResponse]) -> list[TrendPoint]:
    """Derive trend points from recordings (date, severity, classification)."""
    points = []
    for item in recordings:
        rec = item.recording
        res = item.result
        points.append(
            TrendPoint(
                date=rec.created_at,
                severity=res.severity,
                classification=res.classification,
            )
        )
    # Sort by date ascending for trend chart
    points.sort(key=lambda p: p.date)
    return points


@router.get("/patients/{patient_id}/history", response_model=PatientHistoryResponse)
async def get_patient_history(
    patient_id: str,
    days: Optional[int] = Query(default=7, ge=1, le=90, description="Number of days of history"),
):
    """Get recording history and severity trend for a patient.

    Uses real data from the recording cache (populated by the analyze flow).
    Returns empty history when no recordings exist yet.
    """
    items = cache_list_all()
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    recordings: list[RecordingResponse] = []
    for item in items:
        rec_data = item.get("recording", {})
        # Filter by patient_id (use "_all" or "default" for all recordings)
        if patient_id not in ("_all", "default", "") and rec_data.get("patient_id") != patient_id:
            continue
        if rec_data.get("created_at", "") < cutoff:
            continue
        recordings.append(
            RecordingResponse(
                recording=Recording(**rec_data),
                result=ClassificationResult(**item["result"]),
            )
        )

    trend = _build_trend_from_recordings(recordings)
    display_id = "All" if patient_id in ("_all", "default", "") else patient_id[:8]
    patient = Patient(id=patient_id, name=f"Patient {display_id}")

    return PatientHistoryResponse(
        patient=patient,
        recordings=recordings,
        trend=trend,
    )
