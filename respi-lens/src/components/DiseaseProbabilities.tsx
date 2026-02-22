import { motion } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'
import type { DiseaseProbability } from '../types'
import { DISEASE_CLINICAL_NOTES } from '../lib/theme'

interface DiseaseProbabilitiesProps {
    probabilities: DiseaseProbability[]
}

const diseaseColors: Record<string, string> = {
    COPD: '#B45309',
    Pneumonia: '#B91C1C',
    Bronchitis: '#C2410C',
    Asthma: '#0F766E',
    Bronchiectasis: '#047857',
    URTI: '#B45309',
    LRTI: '#B91C1C',
    Healthy: '#047857',
    Bronchiolitis: '#C2410C',
}

export function DiseaseProbabilities({ probabilities }: DiseaseProbabilitiesProps) {
    const sorted = [...probabilities].sort((a, b) => b.probability - a.probability)
    const topTwo = sorted.slice(0, 2)
    const rest = sorted.slice(2)

    return (
        <div className="card-static p-6">
            <h3 className="text-sm font-bold text-[var(--fg)] mb-1">Differential Diagnosis Support</h3>
            <p className="text-[10px] text-[var(--fg-muted)] mb-5 leading-relaxed">
                AI-estimated probabilities — interpret alongside clinical findings
            </p>

            <div className="space-y-4">
                {topTwo.map((dp, i) => {
                    const color = diseaseColors[dp.disease] || 'var(--primary)'
                    const pct = Math.round(dp.probability * 100)
                    const clinicalNote = DISEASE_CLINICAL_NOTES[dp.disease]
                    const isLeading = i === 0

                    return (
                        <motion.div
                            key={dp.disease}
                            initial={{ opacity: 0, x: -12 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.05, duration: 0.2 }}
                            className={`p-3 rounded-[var(--radius)] ${isLeading ? 'bg-[var(--bg-muted)] border border-[var(--border)]' : ''}`}
                        >
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    {isLeading && (
                                        <span
                                            className="w-2 h-2 rounded-full shrink-0"
                                            style={{ backgroundColor: color }}
                                        />
                                    )}
                                    <span className={`text-sm font-medium text-[var(--fg)] ${isLeading ? 'font-semibold' : ''}`}>
                                        {dp.disease}
                                    </span>
                                    {isLeading && (
                                        <span className="text-[9px] font-semibold text-[var(--fg-muted)] bg-[var(--bg-elevated)] px-1.5 py-0.5 rounded">
                                            LEADING
                                        </span>
                                    )}
                                </div>
                                <span className="text-sm font-bold" style={{ color }}>
                                    {pct}%
                                </span>
                            </div>
                            <div className="h-2 bg-[var(--bg-muted)] rounded-[var(--radius-sm)] overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${pct}%` }}
                                    transition={{ duration: 0.4, delay: i * 0.05, ease: 'easeOut' }}
                                    className="h-full rounded-[var(--radius-sm)]"
                                    style={{ backgroundColor: color }}
                                />
                            </div>
                            {clinicalNote && pct >= 15 && (
                                <p className="text-[11px] text-[var(--fg-muted)] mt-2 leading-relaxed italic">
                                    {clinicalNote}
                                </p>
                            )}
                        </motion.div>
                    )
                })}
            </div>

            {rest.length > 0 && (
                <div className="mt-4 pt-4 border-t border-[var(--border)] space-y-3">
                    {rest.map((dp, i) => {
                        const color = diseaseColors[dp.disease] || 'var(--primary)'
                        const pct = Math.round(dp.probability * 100)
                        return (
                            <motion.div
                                key={dp.disease}
                                initial={{ opacity: 0, x: -12 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: (i + 2) * 0.05, duration: 0.2 }}
                            >
                                <div className="flex items-center justify-between mb-1.5">
                                    <span className="text-xs text-[var(--fg-muted)]">{dp.disease}</span>
                                    <span className="text-xs font-semibold" style={{ color }}>
                                        {pct}%
                                    </span>
                                </div>
                                <div className="h-1.5 bg-[var(--bg-muted)] rounded-[var(--radius-sm)] overflow-hidden">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${pct}%` }}
                                        transition={{ duration: 0.4, delay: (i + 2) * 0.05, ease: 'easeOut' }}
                                        className="h-full rounded-[var(--radius-sm)]"
                                        style={{ backgroundColor: color, opacity: 0.7 }}
                                    />
                                </div>
                            </motion.div>
                        )
                    })}
                </div>
            )}

            <div className="mt-5 flex items-start gap-2 text-[10px] text-[var(--fg-subtle)] leading-relaxed">
                <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" strokeWidth={2} />
                <span>
                    Probabilities are AI-estimated and should be interpreted alongside clinical history,
                    physical examination, and additional diagnostic findings.
                </span>
            </div>
        </div>
    )
}
