import { Link } from 'react-router-dom'
import { motion, useInView } from 'framer-motion'
import { useRef, useEffect, useState } from 'react'
import {
    Mic,
    Activity,
    Clock,
    TrendingUp,
    Waves,
    Shield,
    ArrowRight,
    ChevronDown,
} from 'lucide-react'
import { listRecordings } from '../api/client'
import respiBack from '../assets/respi_back.png'
import { RecordingCard, fadeUpVariants } from '../components/RecordingCard'
import type { RecordingWithResult } from '../types'

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
    transition: { duration: 0.25, ease: [0.25, 0.1, 0.25, 1] },
}


export function DashboardPage() {
    const [recentRecordings, setRecentRecordings] = useState<RecordingWithResult[]>([])
    const [stats, setStats] = useState({ count: 0, avgConfidence: 0 })
    const heroRef = useRef(null)
    const statsRef = useRef(null)
    const listRef = useRef(null)
    const [scrollY, setScrollY] = useState(0)

    const heroInView = useInView(heroRef, { once: true, margin: '-50px' })
    const statsInView = useInView(statsRef, { once: true, margin: '-80px' })
    const listInView = useInView(listRef, { once: true, margin: '-80px' })

    useEffect(() => {
        const onScroll = () => setScrollY(window.scrollY)
        onScroll()
        window.addEventListener('scroll', onScroll)
        return () => window.removeEventListener('scroll', onScroll)
    }, [])

    useEffect(() => {
        listRecordings({ limit: 5 })
            .then((data) => {
                setRecentRecordings(data)
                if (data.length > 0) {
                    const avgConf =
                        data.reduce((s, r) => s + r.result.confidence, 0) / data.length
                    const today = new Date().toDateString()
                    const todayCount = data.filter(
                        (r) => new Date(r.recording.created_at).toDateString() === today
                    ).length
                    setStats({
                        count: todayCount,
                        avgConfidence: Math.round(avgConf * 100),
                    })
                }
            })
            .catch(() => setRecentRecordings([]))
    }, [])

    const showScrollIndicator = scrollY < 200

    return (
        <div>
            {/* Hero: Custom illustration background, editorial typography ───────── */}
            <section
                ref={heroRef}
                className="relative min-h-[85vh] flex items-center overflow-hidden"
            >
                {/* Custom respiratory illustration background — centered */}
                <div
                    className="absolute inset-0 pointer-events-none flex items-center justify-center overflow-hidden"
                    aria-hidden
                >
                    <div
                        className="absolute inset-0 min-w-full min-h-full bg-cover bg-center bg-no-repeat opacity-60"
                        style={{ backgroundImage: `url(${respiBack})` }}
                    />
                    <div
                        className="absolute inset-0 bg-[var(--bg)]/75"
                        aria-hidden
                    />
                </div>

                <div className="relative w-full max-w-6xl mx-auto px-6 lg:px-8 py-24 lg:py-32">
                    <div className="relative max-w-4xl pt-16 lg:pt-24">
                        {/* Editorial label with line */}
                        <motion.div
                            initial={{ opacity: 0, x: -8 }}
                            animate={heroInView ? { opacity: 1, x: 0 } : {}}
                            transition={{ delay: 0.1, duration: 0.3 }}
                            className="mb-8"
                        >
                            <span className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.25em] border-l-2 border-[var(--primary)] pl-3">
                                Clinical AI
                            </span>
                        </motion.div>

                        {/* Hero headline — 8xl–9xl */}
                        <h1 className="font-display text-6xl sm:text-7xl lg:text-8xl xl:text-9xl font-normal text-[var(--fg)] leading-[0.95] tracking-tight mb-8">
                            <motion.span
                                initial={{ opacity: 0, y: 24 }}
                                animate={heroInView ? { opacity: 1, y: 0 } : {}}
                                transition={{ delay: 0.15, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                                className="block"
                            >
                                Lung Sound
                            </motion.span>
                            <motion.span
                                initial={{ opacity: 0, y: 24 }}
                                animate={heroInView ? { opacity: 1, y: 0 } : {}}
                                transition={{ delay: 0.22, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                                className="block text-[var(--primary)]"
                            >
                                Analysis
                            </motion.span>
                        </h1>

                        {/* Impact stat — large typographic moment */}
                        <motion.div
                            initial={{ opacity: 0, y: 16 }}
                            animate={heroInView ? { opacity: 1, y: 0 } : {}}
                            transition={{ delay: 0.3, duration: 0.3 }}
                            className="flex items-baseline gap-6 mb-10"
                        >
                            <span className="font-display text-5xl sm:text-6xl text-[var(--primary)]">
                                &lt;10s
                            </span>
                            <span className="text-lg text-[var(--fg-muted)] font-medium">
                                to result
                            </span>
                        </motion.div>

                        <motion.p
                            initial={{ opacity: 0 }}
                            animate={heroInView ? { opacity: 1 } : {}}
                            transition={{ delay: 0.35 }}
                            className="text-lg text-[var(--fg-muted)] max-w-xl leading-relaxed mb-12"
                        >
                            Detect crackles, wheezes, and adventitious sounds with clinical-grade deep learning.
                            Record auscultation audio and get instant classification with SHAP explainability.
                        </motion.p>

                        <motion.div
                            initial={{ opacity: 0, y: 12 }}
                            animate={heroInView ? { opacity: 1, y: 0 } : {}}
                            transition={{ delay: 0.4 }}
                            className="flex flex-wrap gap-4"
                        >
                            <Link
                                to="/record"
                                className="inline-flex items-center gap-2 px-6 py-3.5 bg-[var(--primary)] text-white font-semibold rounded-[var(--radius)] hover:bg-[var(--primary-hover)] transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]"
                            >
                                <Mic className="w-5 h-5" strokeWidth={2} />
                                Start Recording
                                <ArrowRight className="w-4 h-4" strokeWidth={2.5} />
                            </Link>
                            <Link
                                to="/history"
                                className="inline-flex items-center gap-2 px-6 py-3.5 bg-transparent text-[var(--fg)] font-semibold rounded-[var(--radius)] border-2 border-[var(--border-strong)] hover:border-[var(--fg)] transition-all duration-200 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]"
                            >
                                <Clock className="w-5 h-5" strokeWidth={2} />
                                View History
                            </Link>
                        </motion.div>
                    </div>
                </div>

                {/* Scroll indicator */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: showScrollIndicator ? 1 : 0 }}
                    transition={{ duration: 0.2 }}
                    className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
                >
                    <span className="text-[10px] font-semibold text-[var(--fg-subtle)] uppercase tracking-widest">
                        Scroll
                    </span>
                    <motion.div
                        animate={{ y: [0, 6, 0] }}
                        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                    >
                        <ChevronDown className="w-5 h-5 text-[var(--fg-subtle)]" strokeWidth={2} />
                    </motion.div>
                </motion.div>
            </section>

            {/* Stats: Asymmetric layout, full-bleed strip ─────────────────────── */}
            <section
                ref={statsRef}
                className="border-y border-[var(--border)] border-b-[var(--border-accent)] bg-[var(--bg-elevated)]"
            >
                <div className="max-w-6xl mx-auto px-6 lg:px-8 py-12">
                    <motion.div
                        initial="hidden"
                        animate={statsInView ? 'visible' : 'hidden'}
                        variants={{
                            visible: { transition: { staggerChildren: 0.06 } },
                        }}
                        className="grid grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-12"
                    >
                        {/* First stat emphasized */}
                        <motion.div
                            variants={fadeUp}
                            className="lg:col-span-1 flex items-baseline gap-4 group"
                        >
                            <Activity className="w-6 h-6 text-[var(--primary)] shrink-0 mt-1" strokeWidth={2} />
                            <div>
                                <div className="text-3xl lg:text-4xl font-bold text-[var(--fg)] group-hover:text-[var(--primary)] transition-colors duration-200">
                                    {stats.count}
                                </div>
                                <div className="text-sm text-[var(--fg-muted)] mt-1">Recordings Today</div>
                            </div>
                        </motion.div>
                        {[
                            { icon: Waves, label: 'Avg. Confidence', value: recentRecordings.length ? `${stats.avgConfidence}%` : '—' },
                            { icon: TrendingUp, label: 'Trend', value: recentRecordings.length >= 2 ? (recentRecordings[0].result.severity > (recentRecordings[1]?.result.severity ?? 0) ? 'Rising' : 'Stable') : '—' },
                            { icon: Shield, label: 'Model Status', value: 'Active' },
                        ].map((stat) => (
                            <motion.div
                                key={stat.label}
                                variants={fadeUp}
                                className="flex items-baseline gap-4 group"
                            >
                                <stat.icon
                                    className="w-5 h-5 text-[var(--primary)] shrink-0 mt-0.5"
                                    strokeWidth={2}
                                />
                                <div>
                                    <div className="text-2xl font-bold text-[var(--fg)] group-hover:text-[var(--primary)] transition-colors duration-200 font-mono">
                                        {stat.value}
                                    </div>
                                    <div className="text-sm text-[var(--fg-muted)]">{stat.label}</div>
                                </div>
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </section>

            {/* Recent Recordings ─────────────────────────────────────────────── */}
            <section ref={listRef} className="max-w-6xl mx-auto px-6 lg:px-8 py-20 pb-28">
                <motion.div
                    initial={{ opacity: 0, y: 16 }}
                    animate={listInView ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.25 }}
                    className="flex items-end justify-between mb-12"
                >
                    <h2 className="text-2xl font-bold text-[var(--fg)]">Recent Recordings</h2>
                    <Link
                        to="/history"
                        className="text-sm font-semibold text-[var(--primary)] hover:text-[var(--primary-hover)] flex items-center gap-1.5 transition-colors duration-150 link-underline focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)] rounded"
                    >
                        View all
                        <ArrowRight className="w-4 h-4" strokeWidth={2.5} />
                    </Link>
                </motion.div>

                <motion.div
                    initial="hidden"
                    animate={listInView ? 'visible' : 'hidden'}
                    variants={{
                        visible: { transition: { staggerChildren: 0.05 } },
                    }}
                    className="space-y-4"
                >
                    {recentRecordings.length === 0 ? (
                        <p className="text-[var(--fg-muted)] py-8 text-center">
                            No recordings yet. Record your first lung sound analysis to see results here.
                        </p>
                    ) : (
                    recentRecordings.map((recordingWithResult) => (
                        <RecordingCard
                            key={recordingWithResult.recording.id}
                            recordingWithResult={recordingWithResult}
                            variant="dashboard"
                            variants={fadeUpVariants}
                        />
                    ))
                    )}
                </motion.div>
            </section>
        </div>
    )
}
