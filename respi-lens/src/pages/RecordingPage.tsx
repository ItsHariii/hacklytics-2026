import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Mic,
    Square,
    Upload,
    Bluetooth,
    Timer,
    Waves,
    CheckCircle2,
    Loader2,
    AlertCircle,
    FileAudio,
    X,
} from 'lucide-react'
import { useRecordingStore } from '../stores/recordingStore'
import { useAudioRecorder } from '../hooks/useAudioRecorder'
import { Waveform } from '../components/Waveform'
import { analyzeAudio, pollForResult } from '../api/client'

type InputMode = 'microphone' | 'bluetooth' | 'file_upload'
type RecordingStatus = 'idle' | 'recording' | 'processing' | 'complete' | 'error'

export function RecordingPage() {
    const navigate = useNavigate()
    const [inputMode, setInputMode] = useState<InputMode>('microphone')
    const [status, setStatus] = useState<RecordingStatus>('idle')
    const [elapsedTime, setElapsedTime] = useState(0)
    const [dragActive, setDragActive] = useState(false)
    const [uploadedFile, setUploadedFile] = useState<File | null>(null)
    const [bluetoothError, setBtError] = useState<string | null>(null)
    const [errorMessage, setErrorMessage] = useState<string | null>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)
    const timerRef = useRef<number | null>(null)

    const { isRecording, setAnalysisResult, setProcessing, setCurrentRecording, setAudioBlob } = useRecordingStore()
    const { startRecording, stopRecording } = useAudioRecorder()

    useEffect(() => {
        if (isRecording) {
            setElapsedTime(0)
            timerRef.current = window.setInterval(() => {
                setElapsedTime((t) => t + 1)
            }, 1000)
        } else {
            if (timerRef.current) clearInterval(timerRef.current)
        }
        return () => {
            if (timerRef.current) clearInterval(timerRef.current)
        }
    }, [isRecording])

    const formatTime = (seconds: number) => {
        const m = Math.floor(seconds / 60)
        const s = seconds % 60
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    }

    const submitAndPoll = useCallback(
        async (audio: Blob) => {
            setErrorMessage(null)
            setStatus('processing')
            setProcessing(true)

            try {
                const { recording_id } = await analyzeAudio(audio)
                const { recording, result } = await pollForResult(recording_id)

                setAnalysisResult(result)
                setCurrentRecording(recording)
                setProcessing(false)
                setStatus('complete')
                setTimeout(() => navigate(`/analysis/${recording_id}`), 1500)
            } catch (err) {
                setErrorMessage(err instanceof Error ? err.message : 'Analysis failed')
                setProcessing(false)
                setStatus('error')
            }
        },
        [
            setAnalysisResult,
            setCurrentRecording,
            setProcessing,
            navigate,
        ]
    )

    const handleRecord = useCallback(async () => {
        if (isRecording) {
            const blob = await stopRecording()
            if (blob) {
                await submitAndPoll(blob)
            } else {
                setErrorMessage('No audio recorded')
                setStatus('error')
            }
        } else {
            setStatus('recording')
            setErrorMessage(null)
            await startRecording()
        }
    }, [isRecording, startRecording, stopRecording, submitAndPoll])

    const handleFileUpload = useCallback(
        async (file: File) => {
            if (!file.name.match(/\.(wav|webm|mp3|ogg)$/i)) return
            setUploadedFile(file)
            setErrorMessage(null)
            setAudioBlob(null) // Clear any prior mic recording so analysis page uses API audio
            await submitAndPoll(file)
        },
        [submitAndPoll, setAudioBlob]
    )

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault()
            setDragActive(false)
            const file = e.dataTransfer.files[0]
            if (file) handleFileUpload(file)
        },
        [handleFileUpload]
    )

    const handleBluetooth = async () => {
        try {
            setBtError(null)
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const nav = navigator as any
            if (!nav.bluetooth) {
                setBtError('Web Bluetooth is not supported in this browser. Use Chrome or Edge on desktop.')
                return
            }
            await nav.bluetooth.requestDevice({
                filters: [{ services: ['heart_rate'] }],
                optionalServices: ['battery_service'],
            })
            setInputMode('microphone')
            setStatus('idle')
        } catch {
            setBtError('Bluetooth pairing cancelled or device not found.')
        }
    }

    const recordingTips = [
        'Place the stethoscope or microphone firmly against the chest wall',
        'Minimize background noise — quiet rooms yield the best results',
        'Record at least 2–3 full breathing cycles (10–30 seconds)',
        'Avoid talking or moving during the recording',
    ]

    const inputModes = [
        { id: 'microphone' as const, label: 'Microphone', icon: Mic },
        { id: 'bluetooth' as const, label: 'Bluetooth', icon: Bluetooth },
        { id: 'file_upload' as const, label: 'File Upload', icon: Upload },
    ]

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="max-w-3xl mx-auto px-6 lg:px-8 py-12"
        >
            <div className="mb-12">
                <h1 className="text-3xl font-display font-normal text-[var(--fg)] mb-3">
                    Record Auscultation
                </h1>
                <p className="text-[var(--fg-muted)] max-w-xl">
                    Capture lung sounds using your microphone, a Bluetooth stethoscope, or upload a pre-recorded file. Results typically appear within seconds.
                </p>
            </div>

            {/* Input Mode Selector — sharp tabs */}
            <div className="flex gap-1 p-1 bg-[var(--bg-muted)] rounded-[var(--radius)] mb-10">
                {inputModes.map((mode) => (
                    <button
                        key={mode.id}
                        onClick={() => {
                            setInputMode(mode.id)
                            setBtError(null)
                        }}
                        className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-[var(--radius)] text-sm font-semibold transition-all duration-150 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)] ${
                            inputMode === mode.id
                                ? 'bg-[var(--bg-elevated)] text-[var(--fg)] shadow-sm border border-[var(--border)]'
                                : 'text-[var(--fg-muted)] hover:text-[var(--fg)]'
                        }`}
                    >
                        <mode.icon className="w-4 h-4" strokeWidth={2} />
                        <span className="hidden sm:inline">{mode.label}</span>
                    </button>
                ))}
            </div>

            {inputMode === 'microphone' && (
                <div className="space-y-10">
                    <div className="bg-[var(--bg-muted)] border border-[var(--border)] rounded-[var(--radius-lg)] p-6 space-y-8">
                        <div>
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-2">
                                    <Waves className="w-4 h-4 text-[var(--primary)]" strokeWidth={2} />
                                    <span className="text-sm font-semibold text-[var(--fg)]">Live Waveform</span>
                                </div>
                                {isRecording && (
                                    <div className="flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-[var(--destructive)] animate-pulse" />
                                        <span className="text-xs font-mono font-semibold text-[var(--destructive)]">REC</span>
                                    </div>
                                )}
                            </div>
                            <Waveform />
                        </div>

                        <div className="flex justify-center">
                            <div className="flex items-center gap-4 text-3xl font-mono font-bold">
                                <Timer
                                    className={`w-7 h-7 ${isRecording ? 'text-[var(--destructive)]' : 'text-[var(--fg-subtle)]'}`}
                                    strokeWidth={2}
                                />
                                <span className={isRecording ? 'text-[var(--destructive)]' : 'text-[var(--fg-muted)]'}>
                                    {formatTime(elapsedTime)}
                                </span>
                            </div>
                        </div>

                        <div className="flex justify-center">
                            <button
                                onClick={handleRecord}
                                disabled={status === 'processing' || status === 'complete'}
                                className="relative group focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[var(--primary)] rounded-[var(--radius-lg)]"
                            >
                                {isRecording && (
                                    <motion.div
                                        className="absolute inset-0 rounded-[var(--radius-lg)] border-2 border-[var(--destructive)]"
                                        animate={{ scale: [1, 1.15, 1], opacity: [0.4, 0, 0.4] }}
                                        transition={{ duration: 1.2, repeat: Infinity }}
                                        style={{ margin: '-6px' }}
                                    />
                                )}
                                <motion.div
                                    animate={
                                        status === 'idle' && !isRecording
                                            ? { scale: [1, 1.03, 1] }
                                            : {}
                                    }
                                    transition={
                                        status === 'idle'
                                            ? { duration: 2.5, repeat: Infinity, ease: 'easeInOut' }
                                            : {}
                                    }
                                    whileHover={{ scale: status !== 'processing' ? 1.03 : 1 }}
                                    whileTap={{ scale: status !== 'processing' ? 0.97 : 1 }}
                                    className={`w-20 h-20 flex items-center justify-center rounded-[var(--radius-lg)] transition-colors duration-150 ${
                                        isRecording
                                            ? 'bg-[var(--destructive)] text-white'
                                            : status === 'processing'
                                                ? 'bg-[var(--bg-muted)] cursor-not-allowed'
                                                : 'bg-[var(--primary)] text-white hover:bg-[var(--primary-hover)]'
                                    }`}
                                >
                                    {status === 'processing' ? (
                                        <Loader2 className="w-8 h-8 animate-spin" strokeWidth={2.5} />
                                    ) : isRecording ? (
                                        <Square className="w-7 h-7 fill-current" strokeWidth={2} />
                                    ) : (
                                        <Mic className="w-8 h-8" strokeWidth={2} />
                                    )}
                                </motion.div>
                            </button>
                        </div>
                    </div>

                    <AnimatePresence mode="wait">
                        <motion.div
                            key={status}
                            initial={{ opacity: 0, y: 4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -4 }}
                            transition={{ duration: 0.15 }}
                            className="text-center"
                        >
                            {status === 'idle' && (
                                <p className="text-[var(--fg-muted)] text-sm">
                                    Tap the microphone to begin. Aim for 10–30 seconds of clear breathing.
                                </p>
                            )}
                            {status === 'recording' && (
                                <p className="text-[var(--destructive)] text-sm font-semibold">
                                    Recording in progress… Click stop when done.
                                </p>
                            )}
                            {status === 'processing' && (
                                <div className="flex items-center justify-center gap-2 text-[var(--primary)] text-sm font-semibold">
                                    <Loader2 className="w-4 h-4 animate-spin" strokeWidth={2} />
                                    Performing auscultation analysis…
                                </div>
                            )}
                            {status === 'complete' && (
                                <div className="flex items-center justify-center gap-2 text-[var(--success)] text-sm font-semibold">
                                    <CheckCircle2 className="w-4 h-4" strokeWidth={2} />
                                    Classification complete. Preparing clinical report…
                                </div>
                            )}
                            {status === 'error' && errorMessage && (
                                <div className="flex flex-col items-center gap-2 text-[var(--accent)] text-sm font-semibold">
                                    <AlertCircle className="w-4 h-4" strokeWidth={2} />
                                    {errorMessage}
                                </div>
                            )}
                        </motion.div>
                    </AnimatePresence>

                    {status === 'idle' && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.2 }}
                            className="card-static p-5 mt-2"
                        >
                            <p className="text-xs font-semibold text-[var(--fg)] mb-3 uppercase tracking-wider">Recording tips</p>
                            <ul className="space-y-2">
                                {recordingTips.map((tip, i) => (
                                    <li key={i} className="flex items-start gap-2.5 text-sm text-[var(--fg-muted)]">
                                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--primary)] mt-1.5 shrink-0" />
                                        {tip}
                                    </li>
                                ))}
                            </ul>
                        </motion.div>
                    )}
                </div>
            )}

            {inputMode === 'bluetooth' && (
                <div className="card-static p-10 text-center space-y-8">
                    <div className="w-20 h-20 mx-auto bg-[var(--primary)]/15 flex items-center justify-center rounded-[var(--radius-lg)]">
                        <Bluetooth className="w-10 h-10 text-[var(--primary)]" strokeWidth={2} />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-[var(--fg)] mb-2">Pair Digital Stethoscope</h3>
                        <p className="text-[var(--fg-muted)] text-sm max-w-sm mx-auto leading-relaxed">
                            Connect your Eko, Stemoscope, or other Bluetooth-enabled digital stethoscope for
                            high-quality auscultation capture.
                        </p>
                    </div>
                    <button
                        onClick={handleBluetooth}
                        className="inline-flex items-center gap-2 px-6 py-3.5 bg-[var(--primary)] text-white font-semibold rounded-[var(--radius)] hover:bg-[var(--primary-hover)] transition-colors duration-150"
                    >
                        <Bluetooth className="w-5 h-5" strokeWidth={2} />
                        Scan for Devices
                    </button>
                    {bluetoothError && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex items-center gap-2 justify-center text-sm text-[var(--accent)]"
                        >
                            <AlertCircle className="w-4 h-4" strokeWidth={2} />
                            {bluetoothError}
                        </motion.div>
                    )}
                    <p className="text-xs text-[var(--fg-subtle)]">
                        Web Bluetooth requires Chrome or Edge on desktop. Not supported on Safari or Firefox.
                    </p>
                </div>
            )}

            {inputMode === 'file_upload' && (
                <div className="space-y-6">
                    <div
                        className={`drop-zone p-14 text-center cursor-pointer rounded-[var(--radius-lg)] ${
                            dragActive ? 'active' : ''
                        }`}
                        onDragOver={(e) => {
                            e.preventDefault()
                            setDragActive(true)
                        }}
                        onDragLeave={() => setDragActive(false)}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".wav,.webm,.mp3,.ogg"
                            className="hidden"
                            onChange={(e) => {
                                const file = e.target.files?.[0]
                                if (file) {
                                    handleFileUpload(file)
                                    e.target.value = '' // Reset so same file can be re-selected
                                }
                            }}
                        />

                        {status === 'processing' ? (
                            <div className="space-y-4">
                                <Loader2 className="w-12 h-12 text-[var(--primary)] animate-spin mx-auto" strokeWidth={2} />
                                <p className="text-[var(--primary)] font-semibold">Performing auscultation analysis…</p>
                            </div>
                        ) : status === 'complete' ? (
                            <div className="space-y-4">
                                <CheckCircle2 className="w-12 h-12 text-[var(--success)] mx-auto" strokeWidth={2} />
                                <p className="text-[var(--success)] font-semibold">Classification complete. Preparing clinical report…</p>
                            </div>
                        ) : status === 'error' && errorMessage ? (
                            <div className="space-y-4">
                                <AlertCircle className="w-12 h-12 text-[var(--accent)] mx-auto" strokeWidth={2} />
                                <p className="text-[var(--accent)] font-semibold">{errorMessage}</p>
                            </div>
                        ) : uploadedFile ? (
                            <div className="space-y-3">
                                <FileAudio className="w-12 h-12 text-[var(--primary)] mx-auto" strokeWidth={2} />
                                <p className="text-[var(--fg)] font-semibold">{uploadedFile.name}</p>
                                <p className="text-sm text-[var(--fg-muted)]">
                                    {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                                </p>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation()
                                        setUploadedFile(null)
                                    }}
                                    className="inline-flex items-center gap-1 text-sm text-[var(--fg-muted)] hover:text-[var(--fg)] font-medium transition-colors duration-150"
                                >
                                    <X className="w-3.5 h-3.5" strokeWidth={2} />
                                    Remove
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <Upload
                                    className={`w-12 h-12 mx-auto transition-colors duration-150 ${
                                        dragActive ? 'text-[var(--primary)]' : 'text-[var(--fg-subtle)]'
                                    }`}
                                    strokeWidth={2}
                                />
                                <div>
                                    <p className="text-[var(--fg)] font-semibold mb-1">
                                        Drop audio file here or click to browse
                                    </p>
                                    <p className="text-sm text-[var(--fg-muted)]">
                                        Supports .wav, .webm, .mp3, .ogg (max 10MB)
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="card-static p-4 flex items-start gap-3 border-l-2 border-l-[var(--primary)]">
                        <AlertCircle className="w-4 h-4 text-[var(--accent)] mt-0.5 shrink-0" strokeWidth={2} />
                        <p className="text-sm text-[var(--fg-muted)] leading-relaxed">
                            For best results, upload recordings captured at 44.1kHz sample rate with minimal
                            background noise. Recordings from digital stethoscopes (Eko, Littmann, Stemoscope)
                            produce optimal results.
                        </p>
                    </div>
                </div>
            )}
        </motion.div>
    )
}
