// Shared types matching api-contract.md

export interface Recording {
    id: string
    patient_id: string
    audio_url: string
    duration_seconds: number
    input_source: 'microphone' | 'bluetooth' | 'file_upload'
    created_at: string // ISO 8601
}

export interface ClassificationResult {
    id: string
    recording_id: string
    classification: 'normal' | 'crackles' | 'wheezes' | 'both'
    confidence: number // 0.0 – 1.0
    severity: number // 0 – 100
    disease_probabilities: DiseaseProbability[]
    shap_features: ShapFeature[]
    detected_segments: DetectedSegment[]
    respiratory_phase: 'inspiratory' | 'expiratory' | 'both'
    created_at: string
}

export interface DiseaseProbability {
    disease: string
    probability: number // 0.0 – 1.0
}

export interface ShapFeature {
    feature: string
    value: number
    label: string
}

export interface DetectedSegment {
    start_sec: number
    end_sec: number
    sound_type: 'crackles' | 'wheezes' | 'both'
    confidence: number
}

export interface Patient {
    id: string
    name?: string
    created_at: string
}

export interface TrendPoint {
    date: string // ISO 8601
    severity: number
    classification: ClassificationResult['classification']
}

export interface RecordingWithResult {
    recording: Recording
    result: ClassificationResult
}
