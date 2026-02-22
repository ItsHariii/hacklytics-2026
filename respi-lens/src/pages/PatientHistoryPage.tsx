import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Calendar, TrendingUp, Loader2, Mic } from 'lucide-react'
import { getPatientHistory } from '../api/client'
import { TrendChart } from '../components/TrendChart'
import { RecordingCard } from '../components/RecordingCard'
import { useState, useEffect } from 'react'
import type { RecordingWithResult, TrendPoint } from '../types'

export function PatientHistoryPage() {
    const [recordings, setRecordings] = useState<RecordingWithResult[]>([])
    const [trendData, setTrendData] = useState<TrendPoint[]>([])
    const [patientName, setPatientName] = useState<string>('')
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        setLoading(true)
        setError(null)
        getPatientHistory('_all', 7)
            .then((res) => {
                setRecordings(res.recordings)
                setTrendData(res.trend as TrendPoint[])
                setPatientName(res.patient?.name ?? 'All')
            })
            .catch((err) => {
                setError(err instanceof Error ? err.message : 'Failed to load history')
                setRecordings([])
                setTrendData([])
            })
            .finally(() => setLoading(false))
    }, [])

    if (loading) {
        return (
            <div className="max-w-6xl mx-auto px-6 lg:px-8 py-24 text-center">
                <Loader2 className="w-16 h-16 text-[var(--primary)] animate-spin mx-auto mb-6" strokeWidth={2} />
                <h2 className="text-2xl font-bold text-[var(--fg)] mb-2">Loading History</h2>
                <p className="text-[var(--fg-muted)]">Fetching recordings and trends…</p>
            </div>
        )
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="max-w-6xl mx-auto px-6 lg:px-8 py-12"
        >
            <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6 mb-12">
                <div>
                    <h1 className="text-3xl font-display font-normal text-[var(--fg)] mb-2">
                        Patient History
                    </h1>
                    <p className="text-[var(--fg-muted)]">
                        Recording history and longitudinal severity trends for {patientName}.
                    </p>
                </div>
                <div className="flex items-center gap-2 text-sm text-[var(--fg-muted)] font-mono">
                    <Calendar className="w-4 h-4" strokeWidth={2} />
                    <span>Last 7 days · {recordings.length} recordings</span>
                </div>
            </div>

            {error && (
                <div className="mb-6 p-4 rounded-[var(--radius)] bg-red-500/10 text-red-600 dark:text-red-400 text-sm">
                    {error}
                </div>
            )}

            <div className="mb-14">
                <div className="flex items-center gap-2 mb-6">
                    <TrendingUp className="w-5 h-5 text-[var(--primary)]" strokeWidth={2} />
                    <h2 className="text-lg font-bold text-[var(--fg)]">Severity Trend</h2>
                </div>
                {trendData.length > 0 ? (
                    <TrendChart data={trendData} />
                ) : (
                    <div className="card-static p-12 text-center">
                        <TrendingUp className="w-10 h-10 text-[var(--fg-subtle)] mx-auto mb-4" strokeWidth={1.5} />
                        <p className="text-[var(--fg-muted)] mb-2">No trend data yet</p>
                        <p className="text-sm text-[var(--fg-subtle)] mb-6 max-w-sm mx-auto">
                            Record lung sounds over multiple sessions to build a severity trend and track patient progression.
                        </p>
                        <Link
                            to="/record"
                            className="inline-flex items-center gap-2 px-5 py-2.5 bg-[var(--primary)] text-white text-sm font-semibold rounded-[var(--radius)] hover:bg-[var(--primary-hover)] transition-colors duration-150"
                        >
                            <Mic className="w-4 h-4" strokeWidth={2} />
                            Record First Session
                        </Link>
                    </div>
                )}
            </div>

            <div>
                <h2 className="text-xl font-bold text-[var(--fg)] mb-6">All Recordings</h2>
                <div className="space-y-4">
                    {recordings.length === 0 ? (
                        <div className="card-static p-10 text-center">
                            <p className="text-[var(--fg-muted)] mb-2">No recordings found</p>
                            <p className="text-sm text-[var(--fg-subtle)] mb-6 max-w-sm mx-auto">
                                Completed analyses will appear here with full classification details and links to each report.
                            </p>
                            <Link
                                to="/record"
                                className="inline-flex items-center gap-2 px-5 py-2.5 bg-[var(--bg-muted)] text-[var(--fg)] text-sm font-semibold rounded-[var(--radius)] border border-[var(--border)] hover:bg-[var(--border)] transition-colors duration-150"
                            >
                                <Mic className="w-4 h-4" strokeWidth={2} />
                                Start Recording
                            </Link>
                        </div>
                    ) : (
                        recordings.map((recordingWithResult, index) => (
                            <RecordingCard
                                key={recordingWithResult.recording.id}
                                recordingWithResult={recordingWithResult}
                                variant="history"
                                index={index}
                            />
                        ))
                    )}
                </div>
            </div>
        </motion.div>
    )
}
