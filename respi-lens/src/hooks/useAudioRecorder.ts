import { useRef, useCallback } from 'react'
import { useRecordingStore } from '../stores/recordingStore'

export function useAudioRecorder() {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const blobResolveRef = useRef<((blob: Blob) => void) | null>(null)

  const { setRecording, setAudioBlob, setWaveformData } = useRecordingStore()

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      audioContextRef.current = new AudioContext({ sampleRate: 44100 })
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 2048
      
      const source = audioContextRef.current.createMediaStreamSource(stream)
      source.connect(analyserRef.current)

      mediaRecorderRef.current = new MediaRecorder(stream)
      chunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setAudioBlob(blob)
        blobResolveRef.current?.(blob)
        blobResolveRef.current = null
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorderRef.current.start(100)
      setRecording(true)

      // Waveform visualization loop
      const updateWaveform = () => {
        if (!analyserRef.current) return
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
        analyserRef.current.getByteTimeDomainData(dataArray)
        setWaveformData(Array.from(dataArray))
        if (mediaRecorderRef.current?.state === 'recording') {
          requestAnimationFrame(updateWaveform)
        }
      }
      updateWaveform()
    } catch (err) {
      console.error('Failed to start recording:', err)
    }
  }, [setRecording, setAudioBlob, setWaveformData])

  const stopRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      blobResolveRef.current = (blob) => {
        resolve(blob)
      }
      mediaRecorderRef.current?.stop()
      audioContextRef.current?.close()
      setRecording(false)
      // Fallback if onstop never fires (e.g. no data)
      setTimeout(() => {
        if (blobResolveRef.current) {
          blobResolveRef.current = null
          resolve(null)
        }
      }, 1000)
    })
  }, [setRecording])

  return { startRecording, stopRecording }
}
