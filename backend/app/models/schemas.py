"""Pydantic schemas matching the API contract types."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid


# --- Shared Types ---

class DiseaseProbability(BaseModel):
    disease: str
    probability: float = Field(ge=0.0, le=1.0)


class ShapFeature(BaseModel):
    feature: str
    value: float
    label: str


# --- Recording ---

class Recording(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    audio_url: str = ""
    duration_seconds: float = 0.0
    input_source: str = "microphone"  # microphone | bluetooth | file_upload
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# --- Detected Audio Segment ---

class DetectedSegment(BaseModel):
    start_sec: float
    end_sec: float
    sound_type: str  # crackles | wheezes | both
    confidence: float = Field(ge=0.0, le=1.0)


# --- Classification Result ---

class ClassificationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recording_id: str
    classification: str  # normal | crackles | wheezes | both
    confidence: float = Field(ge=0.0, le=1.0)
    severity: int = Field(ge=0, le=100)
    disease_probabilities: List[DiseaseProbability]
    shap_features: List[ShapFeature]
    detected_segments: List[DetectedSegment] = []
    respiratory_phase: str = "inspiratory"  # inspiratory | expiratory | both
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# --- Patient ---

class Patient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# --- Trend Data Point ---

class TrendPoint(BaseModel):
    date: str
    severity: int
    classification: str


# --- API Response Models ---

class AnalyzeResponse(BaseModel):
    recording_id: str
    status: str = "processing"
    message: str = "Audio received. Results will be pushed via Realtime."


class RecordingResponse(BaseModel):
    recording: Recording
    result: ClassificationResult


class PatientHistoryResponse(BaseModel):
    patient: Patient
    recordings: List[RecordingResponse]
    trend: List[TrendPoint]


class ErrorResponse(BaseModel):
    detail: str
