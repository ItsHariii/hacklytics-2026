import { create } from 'zustand'
import type { ClassificationResult, Recording, TrendPoint, RecordingWithResult } from '../types'

interface RecordingState {
  // Recording state
  isRecording: boolean
  audioBlob: Blob | null
  waveformData: number[]
  recordingDuration: number

  // Analysis state
  analysisResult: ClassificationResult | null
  isProcessing: boolean

  // Patient history state
  recordings: RecordingWithResult[]
  trendData: TrendPoint[]

  // Current recording metadata
  currentRecording: Recording | null
  inputSource: 'microphone' | 'bluetooth' | 'file_upload'

  // Actions
  setRecording: (isRecording: boolean) => void
  setAudioBlob: (blob: Blob | null) => void
  setWaveformData: (data: number[]) => void
  setRecordingDuration: (duration: number) => void
  setAnalysisResult: (result: ClassificationResult | null) => void
  setProcessing: (processing: boolean) => void
  setRecordings: (recordings: RecordingWithResult[]) => void
  setTrendData: (data: TrendPoint[]) => void
  setCurrentRecording: (recording: Recording | null) => void
  setInputSource: (source: 'microphone' | 'bluetooth' | 'file_upload') => void
  reset: () => void
}

export const useRecordingStore = create<RecordingState>((set) => ({
  isRecording: false,
  audioBlob: null,
  waveformData: [],
  recordingDuration: 0,
  analysisResult: null,
  isProcessing: false,
  recordings: [],
  trendData: [],
  currentRecording: null,
  inputSource: 'microphone',

  setRecording: (isRecording) => set({ isRecording }),
  setAudioBlob: (audioBlob) => set({ audioBlob }),
  setWaveformData: (waveformData) => set({ waveformData }),
  setRecordingDuration: (recordingDuration) => set({ recordingDuration }),
  setAnalysisResult: (analysisResult) => set({ analysisResult }),
  setProcessing: (isProcessing) => set({ isProcessing }),
  setRecordings: (recordings) => set({ recordings }),
  setTrendData: (trendData) => set({ trendData }),
  setCurrentRecording: (currentRecording) => set({ currentRecording }),
  setInputSource: (inputSource) => set({ inputSource }),
  reset: () =>
    set({
      isRecording: false,
      audioBlob: null,
      waveformData: [],
      recordingDuration: 0,
      analysisResult: null,
      isProcessing: false,
      currentRecording: null,
    }),
}))
