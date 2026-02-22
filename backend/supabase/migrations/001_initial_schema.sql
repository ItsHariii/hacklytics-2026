-- RespiLens Clinical — Initial Database Schema
-- Run this migration in Supabase SQL Editor or via CLI

-- ============================================
-- Enable UUID extension
-- ============================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- patients table
-- ============================================
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE patients IS 'Patient records for RespiLens Clinical';

-- ============================================
-- recordings table
-- ============================================
CREATE TABLE IF NOT EXISTS recordings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    audio_path TEXT NOT NULL,
    duration_seconds FLOAT,
    input_source TEXT NOT NULL CHECK (input_source IN ('microphone', 'bluetooth', 'file_upload')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_recordings_patient_id ON recordings(patient_id);
CREATE INDEX idx_recordings_created_at ON recordings(created_at DESC);

COMMENT ON TABLE recordings IS 'Audio recording metadata — actual files stored in Supabase Storage';

-- ============================================
-- classifications table
-- ============================================
CREATE TABLE IF NOT EXISTS classifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recording_id UUID NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    classification TEXT NOT NULL CHECK (classification IN ('normal', 'crackles', 'wheezes', 'both')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    severity INTEGER NOT NULL CHECK (severity >= 0 AND severity <= 100),
    disease_probs JSONB NOT NULL DEFAULT '[]'::jsonb,
    shap_features JSONB NOT NULL DEFAULT '[]'::jsonb,
    respiratory_phase TEXT NOT NULL CHECK (respiratory_phase IN ('inspiratory', 'expiratory', 'both')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_classifications_recording_id ON classifications(recording_id);

COMMENT ON TABLE classifications IS 'ML classification results with SHAP explainability';

-- ============================================
-- Row Level Security (RLS)
-- ============================================

-- Enable RLS on all tables
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE recordings ENABLE ROW LEVEL SECURITY;
ALTER TABLE classifications ENABLE ROW LEVEL SECURITY;

-- Allow anonymous read/write for demo mode (hackathon)
-- In production, restrict to authenticated users with proper roles
CREATE POLICY "anon_read_patients" ON patients FOR SELECT TO anon USING (true);
CREATE POLICY "anon_insert_patients" ON patients FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_read_recordings" ON recordings FOR SELECT TO anon USING (true);
CREATE POLICY "anon_insert_recordings" ON recordings FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_read_classifications" ON classifications FOR SELECT TO anon USING (true);
CREATE POLICY "anon_insert_classifications" ON classifications FOR INSERT TO anon WITH CHECK (true);

-- Service role has full access (used by backend)
CREATE POLICY "service_all_patients" ON patients FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_recordings" ON recordings FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_classifications" ON classifications FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ============================================
-- Storage bucket (run in Supabase Dashboard > Storage)
-- ============================================
-- Bucket: recordings
-- Access: Private (signed URLs)
-- Retention: 90-day auto-delete (configure via lifecycle policy)
-- Note: Storage bucket creation is typically done via Supabase Dashboard or API, not SQL
