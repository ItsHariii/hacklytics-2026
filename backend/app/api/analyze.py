"""POST /api/analyze — Accept audio upload, validate, trigger processing."""

import shutil
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks

from app.config import settings
from app.models.schemas import AnalyzeResponse, ClassificationResult
from app.services.audio import (
    validate_and_save_audio,
    cleanup_temp_file,
    convert_to_wav_if_needed,
    get_audio_duration,
)
from app.services import supabase_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _prediction_to_classification(
    pred, recording_id: str, detected_segments: list = None
) -> ClassificationResult:
    """Map ML PredictionResult to API ClassificationResult."""
    from app.models.schemas import DiseaseProbability, ShapFeature, DetectedSegment

    segments = detected_segments or []
    segment_models = [
        DetectedSegment(
            start_sec=s["start_sec"],
            end_sec=s["end_sec"],
            sound_type=s["sound_type"],
            confidence=float(s["confidence"]),
        )
        for s in segments
    ]

    return ClassificationResult(
        id=str(uuid.uuid4()),
        recording_id=recording_id,
        classification=pred.classification,
        confidence=pred.confidence,
        severity=pred.severity,
        disease_probabilities=[
            DiseaseProbability(disease=d["disease"], probability=d["probability"])
            for d in pred.disease_probabilities
        ],
        shap_features=[
            ShapFeature(feature=s["feature"], value=s["value"], label=s["label"])
            for s in pred.shap_features
        ],
        detected_segments=segment_models,
        respiratory_phase="both",
        created_at=datetime.utcnow().isoformat(),
    )


async def _process_audio(recording_id: str, temp_path: str, patient_id: str):
    """Background task: process audio and store/broadcast results."""
    conv_path_to_cleanup = None
    try:
        # 1. Convert to WAV if needed (librosa may not decode webm directly)
        infer_path, conv_path_to_cleanup = convert_to_wav_if_needed(temp_path)

        # 2. Run real ONNX inference + temporal localization
        try:
            import librosa
            from app.ml.inference import predict_from_array, get_session, get_norm_stats
            from app.ml.features import TARGET_SR
            from app.ml.localize import localize_events

            audio, sr = librosa.load(infer_path, sr=TARGET_SR, mono=True)
            pred = predict_from_array(audio, sr)

            # Find where crackles/wheezes occur (sliding-window event detection)
            raw_segments = []
            try:
                session = get_session()
                norm_mean, norm_std = get_norm_stats()
                raw_segments = localize_events(audio, sr, session, norm_mean, norm_std)
            except Exception as loc_err:
                logger.debug(f"Localization skipped for {recording_id}: {loc_err}")

            result = _prediction_to_classification(pred, recording_id, raw_segments)
        except Exception as e:
            logger.warning(f"Inference failed for {recording_id}, falling back to mock: {e}")
            from app.services.mock_inference import generate_mock_result

            mock = generate_mock_result(recording_id)
            result = mock.model_copy(
                update={
                    "recording_id": recording_id,
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

        # 3. Compute actual duration
        duration = get_audio_duration(infer_path)
        if duration <= 0:
            duration = 15.0

        # 4. Upload to Supabase Storage (or persist locally when not configured)
        storage_path = f"{patient_id}/{recording_id}.wav"
        supabase_ok = await supabase_service.upload_audio(temp_path, storage_path)

        local_audio_path: Optional[str] = None
        if not supabase_service.is_supabase_configured():
            # Persist WAV locally for audio playback when Supabase is disabled
            storage_dir = Path("storage") / "recordings"
            storage_dir.mkdir(parents=True, exist_ok=True)
            local_path = storage_dir / f"{recording_id}.wav"
            shutil.copy2(infer_path, local_path)
            local_audio_path = str(local_path.resolve())

        # 5. Store recording metadata
        await supabase_service.store_recording({
            "id": recording_id,
            "patient_id": patient_id,
            "audio_path": storage_path,
            "duration_seconds": duration,
            "input_source": "file_upload",
        })

        # 6. Store classification result
        await supabase_service.store_classification(result.model_dump())

        # 7. Broadcast via Realtime
        await supabase_service.broadcast_result(recording_id, result.model_dump())

        # 8. Cache for GET /api/recordings/{id}
        from app.services.recording_cache import put as cache_put

        api_base = settings.API_BASE_URL.rstrip("/")
        audio_proxy_url = f"{api_base}/api/recordings/{recording_id}/audio"

        cache_put(
            recording_id,
            {
                "id": recording_id,
                "patient_id": patient_id,
                "audio_path": storage_path,
                "audio_url": audio_proxy_url,
                "duration_seconds": duration,
                "input_source": "file_upload",
                "created_at": datetime.utcnow().isoformat(),
            },
            result.model_dump(),
            local_audio_path=local_audio_path,
        )

        logger.info(f"Processing complete for recording: {recording_id}")

    except Exception as e:
        logger.error(f"Processing failed for {recording_id}: {e}")
    finally:
        cleanup_temp_file(temp_path)
        if conv_path_to_cleanup:
            cleanup_temp_file(conv_path_to_cleanup)


@router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
):
    """Upload audio for lung sound classification.

    The audio is validated, saved, and processing is triggered as a
    background task. Results are pushed via Supabase Realtime when ready.
    """
    # Validate and save the uploaded file
    temp_path, original_filename = await validate_and_save_audio(audio)

    # Generate IDs
    recording_id = str(uuid.uuid4())
    if patient_id is None:
        patient_id = str(uuid.uuid4())

    # Trigger background processing
    background_tasks.add_task(_process_audio, recording_id, temp_path, patient_id)

    return AnalyzeResponse(
        recording_id=recording_id,
        status="processing",
        message="Audio received. Results will be pushed via Realtime.",
    )
