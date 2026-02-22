import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Activity, ArrowRight } from 'lucide-react'
import { CLASSIFICATION_CONFIG, getSeverityClass, getSeverityLabel, formatDate } from '../lib/theme'
import type { RecordingWithResult } from '../types'

export const fadeUpVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
} as const

interface RecordingCardProps {
    recordingWithResult: RecordingWithResult
    variant?: 'dashboard' | 'history'
    index?: number
    variants?: typeof fadeUpVariants
}

export function RecordingCard({
    recordingWithResult,
    variant = 'dashboard',
    index = 0,
    variants: motionVariants,
}: RecordingCardProps) {
    const { recording, result } = recordingWithResult
    const config = CLASSIFICATION_CONFIG[result.classification]

    const clinicalPreview = `${config.label} · ${result.respiratory_phase ? result.respiratory_phase.charAt(0).toUpperCase() + result.respiratory_phase.slice(1) : 'N/A'} · ${getSeverityLabel(result.severity)} severity (${result.severity}/100)`

    const content = (
        <>
            <div
                className={`shrink-0 w-12 h-12 flex items-center justify-center rounded-[var(--radius)] ${config.bgColor}`}
            >
                <Activity
                    className={`w-6 h-6 ${config.color}`}
                    strokeWidth={2}
                />
            </div>

            <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 mb-1 flex-wrap">
                    <span
                        className={`text-xs font-semibold px-2.5 py-1 rounded-[var(--radius-sm)] ${config.className}`}
                    >
                        {config.label}
                    </span>
                    <span className="text-[10px] text-[var(--fg-subtle)] capitalize">
                        {result.respiratory_phase}
                    </span>
                    {variant === 'dashboard' && (
                        <span className="text-xs text-[var(--fg-subtle)] font-mono">
                            {recording.duration_seconds.toFixed(1)}s
                        </span>
                    )}
                </div>
                <div className="text-sm text-[var(--fg-muted)] font-mono mb-0.5">
                    {variant === 'dashboard'
                        ? `${formatDate(recording.created_at, 'short')} · ${recording.input_source.replace('_', ' ')}`
                        : formatDate(recording.created_at, 'medium')}
                </div>
                {variant === 'history' && (
                    <div className="text-[11px] text-[var(--fg-muted)] italic truncate">
                        {clinicalPreview}
                    </div>
                )}
            </div>

            <div className="hidden sm:flex items-center gap-8">
                {variant === 'history' && (
                    <div className="text-right">
                        <div className="text-xs text-[var(--fg-subtle)] mb-0.5">Duration</div>
                        <div className="text-sm font-semibold text-[var(--fg)] font-mono">
                            {recording.duration_seconds.toFixed(1)}s
                        </div>
                    </div>
                )}
                <div className="text-right">
                    <div className="text-xs text-[var(--fg-subtle)] mb-0.5">Severity</div>
                    <div
                        className={`font-bold font-mono ${variant === 'dashboard' ? 'text-base' : 'text-sm'} ${getSeverityClass(result.severity)}`}
                    >
                        {result.severity}/100
                    </div>
                </div>
                <div className="text-right">
                    <div className="text-xs text-[var(--fg-subtle)] mb-0.5">Confidence</div>
                    <div className={`font-bold font-mono text-[var(--fg)] ${variant === 'dashboard' ? 'text-base' : 'text-sm'}`}>
                        {(result.confidence * 100).toFixed(0)}%
                    </div>
                </div>
                <ArrowRight
                    className="w-5 h-5 text-[var(--fg-subtle)] group-hover:text-[var(--primary)] group-hover:translate-x-0.5 transition-all duration-150"
                    strokeWidth={2}
                />
            </div>
        </>
    )

    return (
        <motion.div
            key={recording.id}
            initial={motionVariants ? 'hidden' : { opacity: 0, y: 12 }}
            animate={motionVariants ? 'visible' : { opacity: 1, y: 0 }}
            transition={
                motionVariants
                    ? { duration: 0.25, ease: [0.25, 0.1, 0.25, 1] }
                    : { delay: index * 0.03, duration: 0.2 }
            }
            variants={motionVariants}
        >
            <Link
                to={`/analysis/${recording.id}`}
                className="card flex items-center gap-6 p-6 group block focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)] rounded-[var(--radius-lg)]"
            >
                {content}
            </Link>
        </motion.div>
    )
}
