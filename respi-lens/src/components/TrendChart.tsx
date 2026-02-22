import { motion } from 'framer-motion'
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
} from 'recharts'
import type { TrendPoint } from '../types'
import { CLASSIFICATION_CONFIG, CLASSIFICATION_HEX } from '../lib/theme'
import type { ClassificationKey } from '../lib/theme'

interface TrendChartProps {
    data: TrendPoint[]
    height?: number
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: TrendPoint }> }) {
    if (!active || !payload?.length) return null
    const point = payload[0].payload
    const date = new Date(point.date).toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
    })

    return (
        <div className="bg-[var(--bg-elevated)] border border-[var(--border)] px-4 py-3 min-w-[160px] rounded-[var(--radius)] shadow-lg">
            <div className="text-xs text-[var(--fg-muted)] mb-1">{date}</div>
            <div className="flex items-center justify-between gap-4">
                <span className="text-sm font-semibold text-[var(--fg)]">Severity</span>
                <span
                    className="text-sm font-bold"
                    style={{ color: CLASSIFICATION_HEX[point.classification as ClassificationKey] }}
                >
                    {point.severity}/100
                </span>
            </div>
            <div
                className="mt-1 text-xs capitalize"
                style={{ color: CLASSIFICATION_HEX[point.classification as ClassificationKey] }}
            >
                {CLASSIFICATION_CONFIG[point.classification as ClassificationKey].label}
            </div>
        </div>
    )
}

export function TrendChart({ data, height = 260 }: TrendChartProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="card-static p-6"
        >
            <div className="flex items-center justify-between mb-5">
                <h3 className="text-sm font-bold text-[var(--fg)]">7-Day Severity Trend</h3>
                <div className="flex items-center gap-4">
                    {(Object.entries(CLASSIFICATION_HEX) as [ClassificationKey, string][]).map(([key, color]) => (
                        <div key={key} className="flex items-center gap-2">
                            <div
                                className="w-2.5 h-2.5 rounded-[var(--radius-sm)]"
                                style={{ backgroundColor: color }}
                            />
                            <span className="text-[10px] font-semibold text-[var(--fg-muted)] uppercase tracking-wider">
                                {key}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            <ResponsiveContainer width="100%" height={height}>
                <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                    <defs>
                        <linearGradient id="severityFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#0F766E" stopOpacity={0.12} />
                            <stop offset="100%" stopColor="#0F766E" stopOpacity={0.12} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis
                        dataKey="date"
                        tickFormatter={(val: string) =>
                            new Date(val).toLocaleDateString('en-US', { weekday: 'short' })
                        }
                        tick={{ fill: 'var(--fg-muted)', fontSize: 11 }}
                        axisLine={{ stroke: 'var(--border)' }}
                        tickLine={false}
                    />
                    <YAxis
                        domain={[0, 100]}
                        tick={{ fill: 'var(--fg-muted)', fontSize: 11 }}
                        axisLine={{ stroke: 'var(--border)' }}
                        tickLine={false}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                        type="monotone"
                        dataKey="severity"
                        stroke="#0F766E"
                        strokeWidth={2}
                        fill="url(#severityFill)"
                        dot={((props: Record<string, unknown>) => {
                            const cx = props.cx as number
                            const cy = props.cy as number
                            const payload = props.payload as TrendPoint
                            return (
                                <circle
                                    key={`${cx}-${cy}`}
                                    cx={cx}
                                    cy={cy}
                                    r={5}
                                    fill={CLASSIFICATION_HEX[payload.classification as ClassificationKey] || '#0F766E'}
                                    stroke="var(--bg-elevated)"
                                    strokeWidth={2}
                                />
                            )
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        }) as any}
                        activeDot={{ r: 7, stroke: '#0F766E', strokeWidth: 2 }}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </motion.div>
    )
}
