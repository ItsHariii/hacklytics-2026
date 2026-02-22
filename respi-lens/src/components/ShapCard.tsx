import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown } from 'lucide-react'
import type { ShapFeature } from '../types'

interface ShapCardProps {
    features: ShapFeature[]
}

export function ShapCard({ features }: ShapCardProps) {
    const supporting = features.filter((f) => f.value > 0)
    const opposing = features.filter((f) => f.value <= 0)
    const maxAbs = Math.max(...features.map((f) => Math.abs(f.value)), 0.01)

    return (
        <div className="card-static p-6">
            <div className="flex items-center justify-between mb-5">
                <h3 className="text-sm font-bold text-[var(--fg)]">Key Acoustic Findings</h3>
                <span className="text-[10px] font-semibold text-[var(--fg-muted)] bg-[var(--bg-muted)] px-2.5 py-1 rounded-[var(--radius-sm)]">
                    Top {features.length}
                </span>
            </div>

            {supporting.length > 0 && (
                <div className="mb-6">
                    <div className="flex items-center gap-2 mb-3">
                        <TrendingUp className="w-3.5 h-3.5 text-[var(--primary)]" strokeWidth={2} />
                        <span className="text-xs font-semibold text-[var(--primary)]">
                            Features supporting this classification
                        </span>
                    </div>
                    <div className="space-y-4">
                        {supporting.map((f, i) => {
                            const barWidth = (Math.abs(f.value) / maxAbs) * 100
                            return (
                                <motion.div
                                    key={f.feature}
                                    initial={{ opacity: 0, y: 8 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.05, duration: 0.2 }}
                                >
                                    <div className="flex items-start justify-between mb-1.5 gap-2">
                                        <span className="text-sm font-medium text-[var(--fg)] leading-snug">
                                            {f.label}
                                        </span>
                                        <span className="text-xs font-bold text-[var(--primary)] shrink-0">
                                            +{f.value.toFixed(3)}
                                        </span>
                                    </div>
                                    <div className="h-2 bg-[var(--bg-muted)] rounded-[var(--radius-sm)] overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${barWidth}%` }}
                                            transition={{ duration: 0.35, delay: i * 0.05 }}
                                            className="h-full rounded-[var(--radius-sm)] bg-[var(--primary)]"
                                        />
                                    </div>
                                    <code className="text-[10px] font-mono text-[var(--fg-subtle)] mt-1 block">
                                        {f.feature}
                                    </code>
                                </motion.div>
                            )
                        })}
                    </div>
                </div>
            )}

            {opposing.length > 0 && (
                <div>
                    <div className="flex items-center gap-2 mb-3">
                        <TrendingDown className="w-3.5 h-3.5 text-[var(--fg-subtle)]" strokeWidth={2} />
                        <span className="text-xs font-semibold text-[var(--fg-subtle)]">
                            Features against this classification
                        </span>
                    </div>
                    <div className="space-y-4">
                        {opposing.map((f, i) => {
                            const barWidth = (Math.abs(f.value) / maxAbs) * 100
                            return (
                                <motion.div
                                    key={f.feature}
                                    initial={{ opacity: 0, y: 8 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: (i + supporting.length) * 0.05, duration: 0.2 }}
                                >
                                    <div className="flex items-start justify-between mb-1.5 gap-2">
                                        <span className="text-sm font-medium text-[var(--fg)] leading-snug">
                                            {f.label}
                                        </span>
                                        <span className="text-xs font-bold text-[var(--fg-subtle)] shrink-0">
                                            {f.value.toFixed(3)}
                                        </span>
                                    </div>
                                    <div className="h-2 bg-[var(--bg-muted)] rounded-[var(--radius-sm)] overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${barWidth}%` }}
                                            transition={{ duration: 0.35, delay: (i + supporting.length) * 0.05 }}
                                            className="h-full rounded-[var(--radius-sm)] bg-[var(--border-strong)]"
                                        />
                                    </div>
                                    <code className="text-[10px] font-mono text-[var(--fg-subtle)] mt-1 block">
                                        {f.feature}
                                    </code>
                                </motion.div>
                            )
                        })}
                    </div>
                </div>
            )}
        </div>
    )
}
