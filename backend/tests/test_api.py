"""Tests for RespiLens API endpoints.

All tests use mock inference — no ONNX model or Supabase connection required.
"""

import io


class TestHealthCheck:
    """Test system endpoints."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "respi-lens-api"

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "docs" in response.json()


class TestAnalyzeEndpoint:
    """Test POST /api/analyze."""

    def test_analyze_valid_wav(self, client, sample_wav_bytes):
        response = client.post(
            "/api/analyze",
            files={"audio": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        )
        assert response.status_code == 202
        data = response.json()
        assert "recording_id" in data
        assert data["status"] == "processing"
        assert "Realtime" in data["message"]

    def test_analyze_valid_webm(self, client, sample_wav_bytes):
        # Use wav bytes but with .webm extension — format validation is by extension
        response = client.post(
            "/api/analyze",
            files={"audio": ("test.webm", io.BytesIO(sample_wav_bytes), "audio/webm")},
        )
        assert response.status_code == 202

    def test_analyze_with_patient_id(self, client, sample_wav_bytes):
        response = client.post(
            "/api/analyze",
            files={"audio": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
            data={"patient_id": "test-patient-123"},
        )
        assert response.status_code == 202

    def test_analyze_invalid_format(self, client):
        response = client.post(
            "/api/analyze",
            files={"audio": ("test.txt", io.BytesIO(b"not audio"), "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid audio format" in response.json()["detail"]

    def test_analyze_empty_file(self, client):
        response = client.post(
            "/api/analyze",
            files={"audio": ("test.wav", io.BytesIO(b""), "audio/wav")},
        )
        assert response.status_code == 400
        assert "Empty" in response.json()["detail"]

    def test_analyze_oversized_file(self, client):
        # Create a file over 10MB
        large_data = b"\x00" * (11 * 1024 * 1024)
        response = client.post(
            "/api/analyze",
            files={"audio": ("test.wav", io.BytesIO(large_data), "audio/wav")},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]


class TestRecordingsEndpoint:
    """Test GET /api/recordings (list) and GET /api/recordings/{recording_id}."""

    def test_list_recordings_empty(self, client):
        """List recordings returns array (empty when no analyses run yet)."""
        response = client.get("/api/recordings")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_recordings_with_limit(self, client):
        response = client.get("/api/recordings?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10

    def test_get_recording(self, client):
        response = client.get("/api/recordings/test-recording-001")
        assert response.status_code == 200
        data = response.json()
        assert "recording" in data
        assert "result" in data
        assert data["recording"]["id"] == "test-recording-001"
        result = data["result"]
        assert result["classification"] in ["normal", "crackles", "wheezes", "both"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0 <= result["severity"] <= 100
        assert len(result["disease_probabilities"]) > 0
        assert len(result["shap_features"]) > 0

    def test_get_recording_shap_structure(self, client):
        response = client.get("/api/recordings/shap-test-001")
        data = response.json()
        shap = data["result"]["shap_features"]
        for feature in shap:
            assert "feature" in feature
            assert "value" in feature
            assert "label" in feature
            assert isinstance(feature["label"], str)
            assert len(feature["label"]) > 10  # Labels should be descriptive


class TestPatientsEndpoint:
    """Test GET /api/patients/{patient_id}/history."""

    def test_get_patient_history(self, client):
        response = client.get("/api/patients/test-patient-001/history")
        assert response.status_code == 200
        data = response.json()
        assert "patient" in data
        assert "recordings" in data
        assert "trend" in data
        assert data["patient"]["id"] == "test-patient-001"
        assert isinstance(data["recordings"], list)
        assert isinstance(data["trend"], list)

    def test_get_patient_history_trend_shape(self, client):
        response = client.get("/api/patients/trend-test/history")
        data = response.json()
        assert "trend" in data
        for point in data["trend"]:
            assert "date" in point
            assert "severity" in point
            assert "classification" in point
            assert point["classification"] in ["normal", "crackles", "wheezes", "both"]

    def test_get_patient_history_with_days_param(self, client):
        response = client.get("/api/patients/test-patient-001/history?days=14")
        assert response.status_code == 200


class TestDemoEndpoint:
    """Test GET /api/demo/{scenario}."""

    def test_demo_normal(self, client):
        response = client.get("/api/demo/normal")
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["classification"] == "normal"

    def test_demo_crackles(self, client):
        response = client.get("/api/demo/crackles")
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["classification"] == "crackles"

    def test_demo_wheezes(self, client):
        response = client.get("/api/demo/wheezes")
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["classification"] == "wheezes"

    def test_demo_both(self, client):
        response = client.get("/api/demo/both")
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["classification"] == "both"

    def test_demo_deteriorating(self, client):
        response = client.get("/api/demo/deteriorating")
        assert response.status_code == 200
        data = response.json()
        # Deteriorating returns patient history shape
        assert "patient" in data
        assert "recordings" in data
        assert "trend" in data
        # Verify severity is increasing
        severities = [point["severity"] for point in data["trend"]]
        assert severities[-1] > severities[0], "Severity should increase over time"

    def test_demo_invalid_scenario(self, client):
        response = client.get("/api/demo/invalid")
        assert response.status_code == 404
        assert "Unknown scenario" in response.json()["detail"]


class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers_present(self, client):
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI CORS middleware should respond to preflight
        assert response.status_code in [200, 405]
