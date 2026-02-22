"""Mock inference service — generates realistic classification results.

This is used until Agent 3 delivers the ONNX model + real inference modules.
All mock data is designed to match the API contract types exactly.
"""

import uuid
import random
from datetime import datetime, timedelta
from typing import List, Optional

from app.models.schemas import (
    ClassificationResult,
    DetectedSegment,
    DiseaseProbability,
    ShapFeature,
    Recording,
    Patient,
    TrendPoint,
    RecordingResponse,
)


# --- Disease probability presets by classification ---

_DISEASE_PRESETS = {
    "normal": [
        DiseaseProbability(disease="Healthy", probability=0.85),
        DiseaseProbability(disease="Asthma", probability=0.06),
        DiseaseProbability(disease="COPD", probability=0.04),
        DiseaseProbability(disease="Pneumonia", probability=0.03),
        DiseaseProbability(disease="Bronchitis", probability=0.02),
    ],
    "crackles": [
        DiseaseProbability(disease="Pneumonia", probability=0.52),
        DiseaseProbability(disease="COPD", probability=0.21),
        DiseaseProbability(disease="Bronchiectasis", probability=0.12),
        DiseaseProbability(disease="LRTI", probability=0.09),
        DiseaseProbability(disease="Healthy", probability=0.06),
    ],
    "wheezes": [
        DiseaseProbability(disease="Asthma", probability=0.48),
        DiseaseProbability(disease="COPD", probability=0.28),
        DiseaseProbability(disease="Bronchitis", probability=0.13),
        DiseaseProbability(disease="URTI", probability=0.07),
        DiseaseProbability(disease="Healthy", probability=0.04),
    ],
    "both": [
        DiseaseProbability(disease="COPD", probability=0.41),
        DiseaseProbability(disease="Pneumonia", probability=0.27),
        DiseaseProbability(disease="Bronchiectasis", probability=0.15),
        DiseaseProbability(disease="Asthma", probability=0.11),
        DiseaseProbability(disease="Healthy", probability=0.06),
    ],
}

# --- SHAP feature presets by classification ---

_SHAP_PRESETS = {
    "normal": [
        ShapFeature(feature="mfcc_1", value=-0.12, label="Low spectral irregularity — consistent with normal breath sounds"),
        ShapFeature(feature="spectral_centroid", value=-0.08, label="Spectral energy centered below 500Hz — no adventitious sounds"),
        ShapFeature(feature="zcr_mean", value=-0.05, label="Low zero-crossing rate — no transient crackle events"),
        ShapFeature(feature="rms_energy", value=0.03, label="Moderate energy levels — typical tidal breathing"),
        ShapFeature(feature="spectral_rolloff", value=-0.02, label="Energy rolloff below 1kHz — expected for normal vesicular sounds"),
    ],
    "crackles": [
        ShapFeature(feature="mfcc_3", value=0.34, label="High spectral energy at 2.3kHz — characteristic fine crackle frequency"),
        ShapFeature(feature="zcr_mean", value=0.28, label="Elevated zero-crossing rate — rapid waveform oscillations from crackles"),
        ShapFeature(feature="spectral_flux", value=0.22, label="High spectral flux — sudden energy bursts typical of discontinuous crackles"),
        ShapFeature(feature="short_time_energy", value=0.19, label="Sharp energy transients — crackles are high-energy, short-duration events"),
        ShapFeature(feature="mfcc_7", value=0.15, label="Fine spectral detail at 3.8kHz — fine crackles above normal breathing range"),
    ],
    "wheezes": [
        ShapFeature(feature="hnr", value=0.41, label="High harmonic-to-noise ratio — wheezes are highly tonal/harmonic"),
        ShapFeature(feature="spectral_centroid", value=0.29, label="Spectral energy shifted to 300-600Hz — wheeze fundamental frequency range"),
        ShapFeature(feature="mfcc_2", value=0.24, label="Strong tonal energy pattern — continuous musical quality of wheezes"),
        ShapFeature(feature="spectral_bandwidth", value=-0.18, label="Narrow spectral bandwidth — wheezes concentrate energy in tight bands"),
        ShapFeature(feature="rms_energy", value=0.12, label="Sustained elevated energy — wheezes persist throughout respiratory phase"),
    ],
    "both": [
        ShapFeature(feature="mfcc_3", value=0.31, label="High spectral energy at 2.3kHz — crackle signature detected"),
        ShapFeature(feature="hnr", value=0.27, label="Elevated harmonic content — wheeze signature overlapping with crackles"),
        ShapFeature(feature="spectral_flux", value=0.23, label="Rapid spectral changes — combined discontinuous and continuous sounds"),
        ShapFeature(feature="zcr_mean", value=0.19, label="High zero-crossing rate — mixed adventitious sound pattern"),
        ShapFeature(feature="spectral_centroid", value=0.16, label="Broadened spectral energy — multiple pathological sound sources"),
    ],
}

_SEVERITY_RANGES = {
    "normal": (5, 15),
    "crackles": (35, 75),
    "wheezes": (30, 65),
    "both": (55, 90),
}

_CONFIDENCE_RANGES = {
    "normal": (0.82, 0.96),
    "crackles": (0.71, 0.92),
    "wheezes": (0.74, 0.90),
    "both": (0.65, 0.85),
}

_SEGMENT_PRESETS = {
    "normal": [],
    "crackles": [
        DetectedSegment(start_sec=2.5, end_sec=4.0, sound_type="crackles", confidence=0.87),
        DetectedSegment(start_sec=7.0, end_sec=8.5, sound_type="crackles", confidence=0.79),
        DetectedSegment(start_sec=11.0, end_sec=12.5, sound_type="crackles", confidence=0.73),
    ],
    "wheezes": [
        DetectedSegment(start_sec=1.5, end_sec=4.0, sound_type="wheezes", confidence=0.88),
        DetectedSegment(start_sec=6.5, end_sec=9.0, sound_type="wheezes", confidence=0.82),
    ],
    "both": [
        DetectedSegment(start_sec=1.0, end_sec=3.5, sound_type="crackles", confidence=0.81),
        DetectedSegment(start_sec=5.0, end_sec=8.0, sound_type="wheezes", confidence=0.76),
        DetectedSegment(start_sec=10.0, end_sec=12.5, sound_type="crackles", confidence=0.84),
        DetectedSegment(start_sec=14.0, end_sec=16.5, sound_type="both", confidence=0.72),
    ],
}


def generate_mock_result(
    recording_id: str,
    classification: Optional[str] = None,
) -> ClassificationResult:
    """Generate a realistic mock classification result."""
    if classification is None:
        classification = random.choice(["normal", "crackles", "wheezes", "both"])

    sev_lo, sev_hi = _SEVERITY_RANGES[classification]
    conf_lo, conf_hi = _CONFIDENCE_RANGES[classification]

    return ClassificationResult(
        id=str(uuid.uuid4()),
        recording_id=recording_id,
        classification=classification,
        confidence=round(random.uniform(conf_lo, conf_hi), 2),
        severity=random.randint(sev_lo, sev_hi),
        disease_probabilities=_DISEASE_PRESETS[classification],
        shap_features=_SHAP_PRESETS[classification],
        detected_segments=_SEGMENT_PRESETS[classification],
        respiratory_phase=random.choice(["inspiratory", "expiratory", "both"]),
    )


# --- Demo Scenario Data ---

_DEMO_PATIENT_ID = "demo-patient-001"
_DEMO_PATIENT = Patient(id=_DEMO_PATIENT_ID, name="Demo Patient")


def get_demo_result(scenario: str) -> Optional[RecordingResponse]:
    """Get a pre-computed demo result for a given scenario."""
    if scenario == "deteriorating":
        return _build_deteriorating_demo()

    if scenario not in _DISEASE_PRESETS:
        return None

    recording_id = str(uuid.uuid4())
    recording = Recording(
        id=recording_id,
        patient_id=_DEMO_PATIENT_ID,
        audio_url=f"https://placeholder.supabase.co/storage/v1/object/recordings/demo/{scenario}.wav",
        duration_seconds=round(random.uniform(8.0, 25.0), 1),
        input_source="file_upload",
    )
    result = generate_mock_result(recording_id, classification=scenario)

    return RecordingResponse(recording=recording, result=result)


def _build_deteriorating_demo() -> RecordingResponse:
    """Build a deteriorating scenario — severity increasing over time."""
    recording_id = str(uuid.uuid4())
    recording = Recording(
        id=recording_id,
        patient_id=_DEMO_PATIENT_ID,
        audio_url=f"https://placeholder.supabase.co/storage/v1/object/recordings/demo/deteriorating.wav",
        duration_seconds=18.5,
        input_source="file_upload",
    )
    result = ClassificationResult(
        id=str(uuid.uuid4()),
        recording_id=recording_id,
        classification="both",
        confidence=0.88,
        severity=78,
        disease_probabilities=_DISEASE_PRESETS["both"],
        shap_features=_SHAP_PRESETS["both"],
        detected_segments=_SEGMENT_PRESETS["both"],
        respiratory_phase="both",
    )
    return RecordingResponse(recording=recording, result=result)


def get_demo_patient_history() -> dict:
    """Build a patient history with a 7-day trend showing deterioration."""
    recordings = []
    trend = []
    now = datetime.utcnow()

    # 7-day trend: normal → crackles → worsening
    progression = [
        ("normal", 12),
        ("normal", 18),
        ("crackles", 35),
        ("crackles", 48),
        ("crackles", 56),
        ("wheezes", 62),
        ("both", 78),
    ]

    for i, (classification, severity) in enumerate(progression):
        day = now - timedelta(days=6 - i)
        rec_id = str(uuid.uuid4())
        recording = Recording(
            id=rec_id,
            patient_id=_DEMO_PATIENT_ID,
            audio_url=f"https://placeholder.supabase.co/storage/v1/object/recordings/demo/day{i+1}.wav",
            duration_seconds=round(random.uniform(10.0, 20.0), 1),
            input_source="file_upload",
            created_at=day.isoformat(),
        )
        result = generate_mock_result(rec_id, classification=classification)
        # Override severity and created_at to match the trend
        result.severity = severity
        result.created_at = day.isoformat()

        recordings.append(RecordingResponse(recording=recording, result=result))
        trend.append(TrendPoint(
            date=day.isoformat(),
            severity=severity,
            classification=classification,
        ))

    return {
        "patient": _DEMO_PATIENT,
        "recordings": recordings,
        "trend": trend,
    }
