"""GET /api/demo/{scenario} — Pre-computed demo results (fallback mode)."""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import RecordingResponse, PatientHistoryResponse
from app.services.mock_inference import get_demo_result, get_demo_patient_history

logger = logging.getLogger(__name__)

router = APIRouter()

VALID_SCENARIOS = {"normal", "crackles", "wheezes", "both", "deteriorating"}


@router.get("/demo/{scenario}")
async def get_demo(scenario: str):
    """Get pre-computed demo results for a given scenario.

    Valid scenarios: normal, crackles, wheezes, both, deteriorating.

    For single classifications (normal/crackles/wheezes/both), returns
    the same shape as GET /api/recordings/{id}.

    For 'deteriorating', returns the same shape as
    GET /api/patients/{id}/history with a 7-day worsening trend.
    """
    if scenario not in VALID_SCENARIOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{scenario}'. Valid scenarios: {', '.join(sorted(VALID_SCENARIOS))}"
        )

    # Deteriorating scenario returns full patient history with trend
    if scenario == "deteriorating":
        history = get_demo_patient_history()
        return PatientHistoryResponse(**history)

    # Single classification scenarios
    result = get_demo_result(scenario)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate demo result")

    return result
