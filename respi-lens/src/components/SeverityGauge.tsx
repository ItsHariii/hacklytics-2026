import { motion } from 'framer-motion'
import { getSeverityLabel, getSeverityColor, getSeverityClinicalGuidance } from '../lib/theme'

interface SeverityGaugeProps {
    severity: number // 0–100
    size?: number
}

const THRESHOLD_ANGLES = [
    { value: 25, label: '25' },
    { value: 50, label: '50' },
    { value: 75, label: '75' },
]

export function SeverityGauge({ severity, size = 160 }: SeverityGaugeProps) {
    const label = getSeverityLabel(severity)
    const color = getSeverityColor(severity)
    const guidance = getSeverityClinicalGuidance(severity)

    const strokeWidth = 8
    const radius = (size - strokeWidth) / 2
    const circumference = 2 * Math.PI * radius
    const arc = circumference * 0.75 // 270° arc
    const offset = arc - (severity / 100) * arc
    const rotation = 135 // Start at bottom-left
    const cx = size / 2
    const cy = size / 2

    return (
        <div className="flex flex-col items-center gap-4">
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
                {/* Background arc */}
                <circle
                    cx={cx}
                    cy={cy}
                    r={radius}
                    fill="none"
                    stroke="var(--border)"
                    strokeWidth={strokeWidth}
                    strokeDasharray={`${arc} ${circumference}`}
                    strokeLinecap="butt"
                    transform={`rotate(${rotation} ${cx} ${cy})`}
                />

                {/* Severity arc */}
                <motion.circle
                    cx={cx}
                    cy={cy}
                    r={radius}
                    fill="none"
                    stroke={color}
                    strokeWidth={strokeWidth}
                    strokeDasharray={`${arc} ${circumference}`}
                    strokeLinecap="butt"
                    transform={`rotate(${rotation} ${cx} ${cy})`}
                    initial={{ strokeDashoffset: arc }}
                    animate={{ strokeDashoffset: offset }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                />

                {/* Threshold tick marks */}
                {THRESHOLD_ANGLES.map((t) => {
                    const angle = rotation + (t.value / 100) * 270
                    const rad = (angle * Math.PI) / 180
                    const outerR = radius + strokeWidth / 2 + 2
                    const innerR = radius - strokeWidth / 2 - 2
                    const labelR = radius + strokeWidth / 2 + 12
                    return (
                        <g key={t.value}>
                            <line
                                x1={cx + innerR * Math.cos(rad)}
                                y1={cy + innerR * Math.sin(rad)}
                                x2={cx + outerR * Math.cos(rad)}
                                y2={cy + outerR * Math.sin(rad)}
                                stroke="var(--fg-subtle)"
                                strokeWidth={1.5}
                            />
                            <text
                                x={cx + labelR * Math.cos(rad)}
                                y={cy + labelR * Math.sin(rad)}
                                textAnchor="middle"
                                dominantBaseline="central"
                                fill="var(--fg-subtle)"
                                fontSize="9"
                                fontWeight="500"
                                style={{ fontFamily: 'inherit' }}
                            >
                                {t.label}
                            </text>
                        </g>
                    )
                })}

                {/* Center text */}
                <text
                    x={cx}
                    y={cy - 6}
                    textAnchor="middle"
                    fill={color}
                    fontSize="28"
                    fontWeight="800"
                    style={{ fontFamily: 'inherit' }}
                >
                    {severity}
                </text>
                <text
                    x={cx}
                    y={cy + 16}
                    textAnchor="middle"
                    fill="var(--fg-muted)"
                    fontSize="11"
                    fontWeight="500"
                    style={{ fontFamily: 'inherit' }}
                >
                    out of 100
                </text>
            </svg>

            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="text-center space-y-2"
            >
                <div className="text-sm font-semibold" style={{ color }}>
                    {label} Severity
                </div>
                <p className="text-xs text-[var(--fg-muted)] leading-relaxed max-w-[240px]">
                    {guidance}
                </p>
            </motion.div>
        </div>
    )
}
