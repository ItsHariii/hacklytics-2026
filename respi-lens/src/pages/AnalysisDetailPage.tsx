import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
    ArrowLeft,
    Download,
    Activity,
    Loader2,
    Stethoscope,
    ShieldCheck,
    ClipboardList,
    AlertTriangle,
    Wind,
} from 'lucide-react'
import { SeverityGauge } from '../components/SeverityGauge'
import { DiseaseProbabilities } from '../components/DiseaseProbabilities'
import { ShapCard } from '../components/ShapCard'
import { AudioEventPlayer } from '../components/AudioEventPlayer'
import {
    CLASSIFICATION_CONFIG,
    formatDate,
    getSeverityLabel,
    getSeverityColor,
    getSeverityClinicalGuidance,
    getConfidenceLevel,
} from '../lib/theme'
import { useRecordingStore } from '../stores/recordingStore'
import { getRecording } from '../api/client'
import type { RecordingWithResult } from '../types'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function handleDownloadPdf(data: RecordingWithResult) {
    const printWindow = window.open('', '_blank')
    if (!printWindow) return

    const { recording, result } = data
    const config = CLASSIFICATION_CONFIG[result.classification as keyof typeof CLASSIFICATION_CONFIG]
        ?? CLASSIFICATION_CONFIG.normal
    const severityLabel = getSeverityLabel(result.severity)
    const severityGuidance = getSeverityClinicalGuidance(result.severity)
    const confLevel = getConfidenceLevel(result.confidence)
    const segments = Array.isArray(result.detected_segments) ? result.detected_segments : []
    const eventCount = segments.length

    printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>RespiLens Clinical Report — ${recording.id}</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 40px; color: #1a1a1a; max-width: 800px; margin: 0 auto; }
        h1 { font-size: 22px; margin-bottom: 4px; }
        h2 { font-size: 16px; margin-top: 28px; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; color: #374151; text-transform: uppercase; letter-spacing: 0.5px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #0F766E; padding-bottom: 16px; margin-bottom: 24px; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: 600; font-size: 13px; }
        .badge.normal { background: #D1FAE5; color: #047857; }
        .badge.crackles { background: #FEF3C7; color: #B45309; }
        .badge.wheezes { background: #FFEDD5; color: #C2410C; }
        .badge.both { background: #FEE2E2; color: #B91C1C; }
        .metric { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f3f4f6; font-size: 13px; }
        .metric-label { color: #6b7280; }
        .metric-value { font-weight: 600; }
        .impression { background: #f9fafb; border-left: 3px solid #0F766E; padding: 12px 16px; margin: 12px 0; font-size: 13px; line-height: 1.6; }
        .actions-list { margin: 8px 0; padding-left: 20px; font-size: 13px; line-height: 1.8; }
        .actions-list li { color: #374151; }
        .shap-item { margin: 6px 0; padding: 8px; background: #f9fafb; border-radius: 4px; font-size: 13px; }
        .shap-name { font-family: monospace; font-size: 11px; color: #9ca3af; }
        .shap-label { font-size: 13px; color: #374151; }
        .shap-value { font-weight: 600; }
        .disclaimer { margin-top: 32px; padding: 12px 16px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 4px; font-size: 11px; color: #92400e; }
        .footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb; color: #9ca3af; font-size: 11px; }
        @media print { body { padding: 20px; } }
      </style>
    </head>
    <body>
      <div class="header">
        <div>
          <h1>RespiLens Clinical Report</h1>
          <p style="color:#6b7280; margin:0; font-size:13px;">AI-Powered Lung Sound Analysis</p>
        </div>
        <div style="text-align:right; color:#6b7280; font-size:12px;">
          <div>Report ID: ${result.id}</div>
          <div>${formatDate(result.created_at, 'long')}</div>
        </div>
      </div>

      <span class="badge ${result.classification}">${config.labelLong}</span>

      <h2>Clinical Impression</h2>
      <div class="impression">${config.clinicalInterpretation}</div>

      <h2>Findings</h2>
      <div class="metric"><span class="metric-label">Classification</span><span class="metric-value">${config.labelLong}</span></div>
      <div class="metric"><span class="metric-label">Confidence</span><span class="metric-value">${(result.confidence * 100).toFixed(1)}% (${confLevel.label})</span></div>
      <div class="metric"><span class="metric-label">Severity Score</span><span class="metric-value">${result.severity}/100 — ${severityLabel}</span></div>
      <div class="metric"><span class="metric-label">Respiratory Phase</span><span class="metric-value" style="text-transform:capitalize">${result.respiratory_phase}</span></div>
      <div class="metric"><span class="metric-label">Recording Duration</span><span class="metric-value">${recording.duration_seconds.toFixed(1)}s</span></div>
      <div class="metric"><span class="metric-label">Detected Events</span><span class="metric-value">${eventCount} adventitious sound event${eventCount !== 1 ? 's' : ''}</span></div>

      <p style="font-size:12px; color:#6b7280; margin-top:8px; line-height:1.5;">${severityGuidance}</p>

      <h2>Differential Diagnosis Support</h2>
      ${result.disease_probabilities
            .sort((a, b) => b.probability - a.probability)
            .map((dp) => `<div class="metric"><span class="metric-label">${dp.disease}</span><span class="metric-value">${(dp.probability * 100).toFixed(1)}%</span></div>`)
            .join('')}
      <p style="font-size:11px; color:#9ca3af; margin-top:6px;">Probabilities are AI-estimated and should be interpreted alongside clinical findings.</p>

      <h2>Key Acoustic Findings</h2>
      ${result.shap_features
            .map(
                (f) => `<div class="shap-item"><span class="shap-label">${f.label}</span> <span class="shap-value" style="color:${f.value > 0 ? '#0F766E' : '#6b7280'}">${f.value > 0 ? '+' : ''}${f.value.toFixed(3)}</span><br/><span class="shap-name">${f.feature}</span></div>`
            )
            .join('')}

      <h2>Recommended Actions</h2>
      <ul class="actions-list">
        ${config.suggestedActions.map((a) => `<li>${a}</li>`).join('')}
      </ul>

      <div class="disclaimer">
        <strong>Disclaimer:</strong> This report is generated by an AI-powered lung sound analysis system and is intended to assist clinical decision-making. It should not replace professional medical judgment, physical examination, or additional diagnostic testing.
      </div>

      <div class="footer">
        <p>Generated by RespiLens Clinical · AI-Powered Lung Sound Analysis</p>
      </div>
    </body>
    </html>
  `)
    printWindow.document.close()
    setTimeout(() => printWindow.print(), 500)
}

export function AnalysisDetailPage() {
    const { id } = useParams<{ id: string }>()
    const { analysisResult, currentRecording, audioBlob } = useRecordingStore()
    const [data, setData] = useState<RecordingWithResult | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [objectUrl, setObjectUrl] = useState<string | null>(null)

    useEffect(() => {
        if (!id) {
            setLoading(false)
            setData(null)
            return
        }

        const stored = currentRecording && analysisResult && currentRecording.id === id
        if (stored) {
            setData({ recording: currentRecording, result: analysisResult })
            setLoading(false)
            setError(null)
            return
        }

        let cancelled = false
        setLoading(true)
        setError(null)

        getRecording(id)
            .then((res) => {
                if (!cancelled) {
                    setData(res)
                    setError(null)
                }
            })
            .catch((err) => {
                if (!cancelled) {
                    setData(null)
                    setError(err instanceof Error ? err.message : 'Failed to load')
                }
            })
            .finally(() => {
                if (!cancelled) setLoading(false)
            })

        return () => {
            cancelled = true
        }
    }, [id, currentRecording, analysisResult])

    // Only use audioBlob for microphone recordings — for file uploads we must fetch from API
    // to avoid playing a stale blob from a previous mic recording
    const useBlob = !!(
        id &&
        audioBlob &&
        currentRecording?.id === id &&
        currentRecording?.input_source === 'microphone'
    )
    useEffect(() => {
        if (!useBlob || !audioBlob) {
            setObjectUrl(null)
            return
        }
        const url = URL.createObjectURL(audioBlob)
        setObjectUrl(url)
        return () => {
            URL.revokeObjectURL(url)
            setObjectUrl(null)
        }
    }, [useBlob, audioBlob])

    if (loading) {
        return (
            <div className="max-w-3xl mx-auto px-6 py-24 text-center">
                <Loader2 className="w-16 h-16 text-[var(--primary)] animate-spin mx-auto mb-6" strokeWidth={2} />
                <h2 className="text-2xl font-bold text-[var(--fg)] mb-2">Loading Analysis</h2>
                <p className="text-[var(--fg-muted)]">Fetching recording results…</p>
            </div>
        )
    }

    if (error || !data) {
        return (
            <div className="max-w-3xl mx-auto px-6 py-24 text-center">
                <Activity className="w-16 h-16 text-[var(--fg-subtle)] mx-auto mb-6" strokeWidth={2} />
                <h2 className="text-2xl font-bold text-[var(--fg)] mb-2">Recording Not Found</h2>
                <p className="text-[var(--fg-muted)] mb-8">
                    {error ?? "The requested recording doesn't exist or has expired."}
                </p>
                <div className="flex flex-wrap justify-center gap-4">
                    <Link
                        to="/record"
                        className="inline-flex items-center gap-2 px-6 py-3 bg-[var(--primary)] text-white font-semibold rounded-[var(--radius)] hover:bg-[var(--primary-hover)] transition-colors duration-150"
                    >
                        Record New
                    </Link>
                    <Link
                        to="/history"
                        className="inline-flex items-center gap-2 px-6 py-3 bg-[var(--bg-muted)] text-[var(--fg)] font-semibold rounded-[var(--radius)] hover:bg-[var(--border)] transition-colors duration-150"
                    >
                        <ArrowLeft className="w-4 h-4" strokeWidth={2} />
                        Back to History
                    </Link>
                </div>
            </div>
        )
    }

    const { recording, result } = data
    const config = CLASSIFICATION_CONFIG[result.classification as keyof typeof CLASSIFICATION_CONFIG]
        ?? CLASSIFICATION_CONFIG.normal
    const confLevel = getConfidenceLevel(result.confidence)
    const severityLabel = getSeverityLabel(result.severity)
    const severityColor = getSeverityColor(result.severity)
    const segments = Array.isArray(result?.detected_segments) ? result.detected_segments : []
    const eventCount = segments.length

    const effectiveAudioUrl = objectUrl ?? (id ? `${API_BASE}/api/recordings/${id}/audio` : '')

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="max-w-6xl mx-auto px-6 lg:px-8 py-12"
        >
            {/* Top bar */}
            <div className="flex items-center justify-between mb-10">
                <Link
                    to="/history"
                    className="flex items-center gap-2 text-sm font-semibold text-[var(--fg-muted)] hover:text-[var(--fg)] transition-colors duration-150 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)] rounded"
                >
                    <ArrowLeft className="w-4 h-4" strokeWidth={2} />
                    Back to History
                </Link>
                <div className="flex items-center gap-3">
                    <Link
                        to="/record"
                        className="inline-flex items-center gap-2 px-5 py-2.5 bg-[var(--bg-muted)] text-[var(--fg)] text-sm font-semibold rounded-[var(--radius)] border border-[var(--border)] hover:bg-[var(--border)] transition-colors duration-150 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]"
                    >
                        Record New
                    </Link>
                    <button
                        onClick={() => handleDownloadPdf({ recording, result })}
                        className="inline-flex items-center gap-2 px-5 py-2.5 bg-[var(--primary)] text-white text-sm font-semibold rounded-[var(--radius)] hover:bg-[var(--primary-hover)] transition-colors duration-150 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                    >
                        <Download className="w-4 h-4" strokeWidth={2} />
                        Download Report
                    </button>
                </div>
            </div>

            {/* A. Clinical Summary Banner */}
            <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className={`card-static p-8 mb-8 border-l-4 ${config.borderColor}`}
            >
                <div className="flex flex-col sm:flex-row sm:items-start gap-6">
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: 0.15, duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                        className={`w-14 h-14 rounded-[var(--radius)] ${config.bgColor} flex items-center justify-center shrink-0`}
                    >
                        <Activity className={`w-7 h-7 ${config.color}`} strokeWidth={2} />
                    </motion.div>
                    <div className="flex-1">
                        <motion.div
                            initial={{ opacity: 0, y: 12 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                            className="flex items-center gap-3 flex-wrap mb-2"
                        >
                            <h1 className={`text-2xl sm:text-3xl font-bold ${config.color}`}>
                                {config.label}
                            </h1>
                            <span className={`text-xs font-semibold px-3 py-1.5 rounded-[var(--radius-sm)] ${config.className}`}>
                                {confLevel.label} · {(result.confidence * 100).toFixed(0)}%
                            </span>
                        </motion.div>
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.25, duration: 0.3 }}
                        >
                            <p className="text-sm text-[var(--fg)] leading-relaxed mb-2 font-medium">
                                Clinical Impression
                            </p>
                            <p className="text-[var(--fg-muted)] text-sm leading-relaxed max-w-3xl">
                                {config.clinicalInterpretation}
                            </p>
                            <p className="text-xs text-[var(--fg-subtle)] mt-2 italic">
                                {confLevel.guidance}
                            </p>
                        </motion.div>
                    </div>
                </div>
            </motion.div>

            {/* B. Clinical Assessment — 2-column layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {/* Left: Auscultation Summary */}
                <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="card-static p-6"
                >
                    <div className="flex items-center gap-2 mb-4">
                        <Stethoscope className="w-4 h-4 text-[var(--primary)]" strokeWidth={2} />
                        <h3 className="text-sm font-bold text-[var(--fg)]">Auscultation Summary</h3>
                    </div>
                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-[var(--fg-muted)]">Classification</span>
                            <span className={`text-sm font-semibold ${config.color}`}>{config.labelLong}</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-[var(--fg-muted)]">Respiratory Phase</span>
                            <span className="text-sm font-semibold text-[var(--fg)] capitalize">{result.respiratory_phase}</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-[var(--fg-muted)]">Detected Events</span>
                            <span className="text-sm font-semibold text-[var(--fg)]">
                                {eventCount} adventitious sound{eventCount !== 1 ? 's' : ''}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-[var(--fg-muted)]">Duration</span>
                            <span className="text-sm font-semibold text-[var(--fg)] font-mono">{recording.duration_seconds.toFixed(1)}s</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-[var(--fg-muted)]">Input Source</span>
                            <span className="text-sm text-[var(--fg)] capitalize">{recording.input_source.replace('_', ' ')}</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-[var(--fg-muted)]">Recorded</span>
                            <span className="text-sm text-[var(--fg)] font-mono">{formatDate(recording.created_at, 'short')}</span>
                        </div>
                    </div>
                </motion.div>

                {/* Right: Risk Assessment */}
                <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.15 }}
                    className="card-static p-6"
                >
                    <div className="flex items-center gap-2 mb-4">
                        <ShieldCheck className="w-4 h-4 text-[var(--primary)]" strokeWidth={2} />
                        <h3 className="text-sm font-bold text-[var(--fg)]">Risk Assessment</h3>
                    </div>
                    <div className="flex justify-center mb-2">
                        <SeverityGauge severity={result.severity} size={160} />
                    </div>
                    <div className="mt-4 space-y-3 border-t border-[var(--border)] pt-4">
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-[var(--fg-muted)]">Risk Level</span>
                            <span className="text-sm font-bold" style={{ color: severityColor }}>
                                {severityLabel}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-[var(--fg-muted)]">Confidence</span>
                            <span className="text-sm font-semibold text-[var(--fg)]">
                                {confLevel.label}
                            </span>
                        </div>
                    </div>
                </motion.div>
            </div>

            {/* Clinical Notes */}
            <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="card-static p-6 mb-8 border-l-2 border-l-[var(--primary)]"
            >
                <div className="flex items-center gap-2 mb-3">
                    <Wind className="w-4 h-4 text-[var(--primary)]" strokeWidth={2} />
                    <h3 className="text-sm font-bold text-[var(--fg)]">Clinical Notes</h3>
                </div>
                <p className="text-sm text-[var(--fg-muted)] leading-relaxed">
                    {config.description} Respiratory phase: <span className="capitalize font-medium">{result.respiratory_phase}</span>.
                    Severity scored at {result.severity}/100 ({severityLabel.toLowerCase()}).
                    {eventCount > 0 && ` ${eventCount} distinct adventitious sound event${eventCount !== 1 ? 's' : ''} identified in the recording.`}
                    {' '}{confLevel.guidance}
                </p>
            </motion.div>

            {/* Audio Player */}
            <AudioEventPlayer
                audioUrl={effectiveAudioUrl}
                durationSeconds={recording.duration_seconds}
                segments={segments}
                classification={result.classification ?? 'normal'}
            />

            {/* Disease Probabilities + SHAP — 2-column */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                <DiseaseProbabilities probabilities={result.disease_probabilities} />
                <ShapCard features={result.shap_features} />
            </div>

            {/* F. Suggested Actions */}
            <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="card-static p-6 mb-6"
            >
                <div className="flex items-center gap-2 mb-4">
                    <ClipboardList className="w-4 h-4 text-[var(--primary)]" strokeWidth={2} />
                    <h3 className="text-sm font-bold text-[var(--fg)]">Suggested Actions</h3>
                </div>
                <ul className="space-y-2.5">
                    {config.suggestedActions.map((action, i) => (
                        <motion.li
                            key={i}
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.35 + i * 0.05, duration: 0.2 }}
                            className="flex items-start gap-3"
                        >
                            <span
                                className="mt-1.5 w-1.5 h-1.5 rounded-full shrink-0"
                                style={{ backgroundColor: severityColor }}
                            />
                            <span className="text-sm text-[var(--fg-muted)] leading-relaxed">{action}</span>
                        </motion.li>
                    ))}
                </ul>
                <div className="mt-5 flex items-start gap-2 text-[10px] text-[var(--fg-subtle)] leading-relaxed">
                    <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" strokeWidth={2} />
                    <span>
                        AI-assisted suggestions are intended to support clinical decision-making. They do not constitute
                        medical advice and should not replace professional judgment.
                    </span>
                </div>
            </motion.div>

            {/* Bottom actions */}
            <div className="flex flex-wrap items-center justify-between gap-4 mt-2 pt-6 border-t border-[var(--border)]">
                <Link
                    to="/history"
                    className="flex items-center gap-2 text-sm font-semibold text-[var(--fg-muted)] hover:text-[var(--fg)] transition-colors duration-150"
                >
                    <ArrowLeft className="w-4 h-4" strokeWidth={2} />
                    All Recordings
                </Link>
                <Link
                    to="/record"
                    className="inline-flex items-center gap-2 px-5 py-2.5 bg-[var(--primary)] text-white text-sm font-semibold rounded-[var(--radius)] hover:bg-[var(--primary-hover)] transition-colors duration-150"
                >
                    Record Another
                </Link>
            </div>
        </motion.div>
    )
}
