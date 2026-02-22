"""Pytest fixtures for RespiLens API tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.recording_cache import put as cache_put


def _seed_recording_cache():
    """Seed cache with test recordings for GET /api/recordings/{id} tests."""
    cache_put(
        "test-recording-001",
        {
            "id": "test-recording-001",
            "patient_id": "test-patient",
            "audio_url": "https://example.com/audio.wav",
            "duration_seconds": 15.0,
            "input_source": "microphone",
            "created_at": "2025-01-15T10:00:00",
        },
        {
            "id": "res-001",
            "recording_id": "test-recording-001",
            "classification": "normal",
            "confidence": 0.89,
            "severity": 12,
            "disease_probabilities": [{"disease": "Healthy", "probability": 0.85}],
            "shap_features": [
                {"feature": "mfcc_1", "value": -0.12, "label": "Low spectral irregularity"},
            ],
            "detected_segments": [],
            "respiratory_phase": "inspiratory",
            "created_at": "2025-01-15T10:00:00",
        },
    )
    cache_put(
        "shap-test-001",
        {
            "id": "shap-test-001",
            "patient_id": "test-patient",
            "audio_url": "https://example.com/audio2.wav",
            "duration_seconds": 18.0,
            "input_source": "file_upload",
            "created_at": "2025-01-15T11:00:00",
        },
        {
            "id": "res-002",
            "recording_id": "shap-test-001",
            "classification": "crackles",
            "confidence": 0.82,
            "severity": 45,
            "disease_probabilities": [{"disease": "Pneumonia", "probability": 0.52}],
            "shap_features": [
                {"feature": "mfcc_3", "value": 0.34, "label": "High spectral energy at 2.3kHz — characteristic fine crackle frequency"},
                {"feature": "zcr_mean", "value": 0.28, "label": "Elevated zero-crossing rate — rapid waveform oscillations"},
            ],
            "detected_segments": [],
            "respiratory_phase": "both",
            "created_at": "2025-01-15T11:00:00",
        },
    )


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    _seed_recording_cache()
    return TestClient(app)


@pytest.fixture
def sample_wav_bytes():
    """Generate minimal valid WAV file bytes for testing."""
    # Minimal WAV header (44 bytes) + 100 bytes of silence
    import struct
    num_samples = 100
    sample_rate = 44100
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    file_size = 36 + data_size

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', file_size, b'WAVE',
        b'fmt ', 16,  # PCM format chunk
        1,  # PCM format
        num_channels, sample_rate, byte_rate,
        block_align, bits_per_sample,
        b'data', data_size,
    )
    audio_data = b'\x00\x00' * num_samples  # silence
    return header + audio_data
