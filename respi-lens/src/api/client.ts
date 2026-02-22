/**
 * API client for RespiLens backend.
 * Uses VITE_API_URL from environment (default: http://localhost:8000).
 */

import type { RecordingWithResult } from '../types'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export interface AnalyzeResponse {
  recording_id: string
  status: string
  message: string
}

/**
 * Upload audio for analysis. Returns recording_id for polling.
 */
export async function analyzeAudio(
  audio: Blob | File,
  patientId?: string
): Promise<AnalyzeResponse> {
  const formData = new FormData()
  const filename = audio instanceof File ? audio.name : (audio.type === 'audio/webm' ? 'recording.webm' : 'recording.wav')
  formData.append('audio', audio, filename)
  if (patientId) {
    formData.append('patient_id', patientId)
  }

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? `Analysis failed: ${res.status}`)
  }

  return res.json()
}

/**
 * Fetch recording and classification result by ID.
 */
export async function getRecording(recordingId: string): Promise<RecordingWithResult> {
  const res = await fetch(`${API_BASE}/api/recordings/${recordingId}`)

  if (!res.ok) {
    if (res.status === 404) {
      throw new Error('Recording not found')
    }
    throw new Error(`Failed to fetch recording: ${res.status}`)
  }

  return res.json()
}

/**
 * List recordings from the backend (real model results).
 */
export async function listRecordings(
  options?: { limit?: number; patient_id?: string }
): Promise<RecordingWithResult[]> {
  const params = new URLSearchParams()
  if (options?.limit) params.set('limit', String(options.limit))
  if (options?.patient_id) params.set('patient_id', options.patient_id)
  const qs = params.toString()
  const url = `${API_BASE}/api/recordings${qs ? `?${qs}` : ''}`

  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to list recordings: ${res.status}`)
  }
  const data = await res.json()
  return Array.isArray(data) ? data : []
}

/**
 * Get patient history (recordings + trend).
 */
export async function getPatientHistory(
  patientId: string,
  days?: number
): Promise<{ patient: { id: string; name?: string }; recordings: RecordingWithResult[]; trend: { date: string; severity: number; classification: string }[] }> {
  const params = new URLSearchParams()
  if (days) params.set('days', String(days))
  const qs = params.toString()
  const url = `${API_BASE}/api/patients/${encodeURIComponent(patientId)}/history${qs ? `?${qs}` : ''}`

  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch patient history: ${res.status}`)
  }
  return res.json()
}

/**
 * Poll for recording result until ready or timeout.
 */
export async function pollForResult(
  recordingId: string,
  options: { intervalMs?: number; timeoutMs?: number } = {}
): Promise<RecordingWithResult> {
  const { intervalMs = 1500, timeoutMs = 45000 } = options
  const start = Date.now()

  while (Date.now() - start < timeoutMs) {
    try {
      const data = await getRecording(recordingId)
      // If we get a valid response with result, we're done
      if (data?.recording?.id && data?.result?.recording_id) {
        return data
      }
    } catch {
      // Keep polling
    }
    await new Promise((r) => setTimeout(r, intervalMs))
  }

  throw new Error('Analysis timed out. Please try again.')
}
