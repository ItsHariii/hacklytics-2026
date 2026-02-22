import { motion } from 'framer-motion'
import { useRecordingStore } from '../stores/recordingStore'
import { CLASSIFICATION_CONFIG, getSeverityColor } from '../lib/theme'

export function AnalysisCard() {
    const { analysisResult } = useRecordingStore()

    if (!analysisResult) return null

    const severityColor = getSeverityColor(analysisResult.severity)

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="card-static p-6"
        >
            <h3 className="text-base font-bold text-[var(--fg)] mb-4">Analysis Result</h3>

            <div
                className={`text-xl font-bold mb-2 ${CLASSIFICATION_CONFIG[analysisResult.classification].color}`}
            >
                {CLASSIFICATION_CONFIG[analysisResult.classification].labelLong}
            </div>

            <div className="text-[var(--fg-muted)] mb-4">
                Confidence: {(analysisResult.confidence * 100).toFixed(1)}%
            </div>

            <div className="mb-4">
                <div className="text-sm text-[var(--fg-muted)] mb-2">Severity Score</div>
                <div className="h-2 bg-[var(--bg-muted)] rounded-[var(--radius-sm)] overflow-hidden">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${analysisResult.severity}%` }}
                        transition={{ duration: 0.4, ease: 'easeOut' }}
                        className="h-full rounded-[var(--radius-sm)]"
                        style={{ backgroundColor: severityColor }}
                    />
                </div>
                <div className="text-right text-sm text-[var(--fg-muted)] mt-1">
                    {analysisResult.severity}/100
                </div>
            </div>

            <div>
                <div className="text-sm font-semibold text-[var(--fg-muted)] mb-2">
                    Key Features (SHAP)
                </div>
                <div className="space-y-2">
                    {analysisResult.shap_features.slice(0, 5).map((f, i) => (
                        <div key={i} className="flex justify-between text-sm">
                            <span className="text-[var(--fg)]">{f.label}</span>
                            <span
                                className={
                                    f.value > 0
                                        ? 'text-[var(--primary)] font-semibold'
                                        : 'text-[var(--fg-subtle)]'
                                }
                            >
                                {f.value > 0 ? '+' : ''}
                                {f.value.toFixed(3)}
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </motion.div>
    )
}
