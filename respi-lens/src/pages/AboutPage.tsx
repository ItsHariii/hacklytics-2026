import { Link } from 'react-router-dom'
import { motion, useInView } from 'framer-motion'
import { useRef } from 'react'
import {
    Mic,
    Activity,
    TrendingUp,
    BarChart3,
    LineChart,
    FileText,
    ArrowRight,
} from 'lucide-react'
import { aboutContent } from '../data/aboutContent'
import respiBack from '../assets/respi_back.png'
import { AnimatedSection } from '../components/AnimatedSection'

const featureIcons: Record<string, React.ComponentType<{ className?: string; strokeWidth?: number }>> = {
    Mic,
    Activity,
    TrendingUp,
    BarChart: BarChart3,
    LineChart,
    FileText,
}

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
}

export function AboutPage() {
    const heroRef = useRef(null)
    const heroInView = useInView(heroRef, { once: true, margin: '-50px' })

    return (
        <div>
            {/* Hero */}
            <section ref={heroRef} className="relative min-h-[70vh] flex items-center overflow-hidden">
                <div className="absolute inset-0 pointer-events-none">
                    <div
                        className="absolute inset-0 min-w-full min-h-full bg-cover bg-center bg-no-repeat opacity-[0.35]"
                        style={{ backgroundImage: `url(${respiBack})` }}
                    />
                    <div className="absolute inset-0 bg-[var(--bg)]/80" />
                </div>
                <div className="relative w-full max-w-4xl mx-auto px-6 lg:px-8 py-24 lg:py-32">
                    <motion.div
                        initial={{ opacity: 0, y: 24 }}
                        animate={heroInView ? { opacity: 1, y: 0 } : {}}
                        transition={{ duration: 0.4 }}
                        className="max-w-3xl"
                    >
                        <p className="text-sm font-semibold text-[var(--primary)] mb-4 italic">
                            {aboutContent.hero.tagline}
                        </p>
                        <span className="inline-block w-12 h-px bg-[var(--primary)] mb-6" />
                        <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-normal text-[var(--fg)] leading-[1.1] tracking-tight mb-6">
                            {aboutContent.hero.headline}
                        </h1>
                        <p className="text-lg text-[var(--fg-muted)] leading-relaxed max-w-2xl">
                            {aboutContent.hero.subheadline}
                        </p>
                    </motion.div>
                </div>
            </section>

            {/* The Problem */}
            <AnimatedSection
                className="py-20 lg:py-28 border-t border-[var(--border)]"
                contentClassName="max-w-4xl mx-auto px-6 lg:px-8"
                staggerChildren={0.08}
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            01
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-8">
                            {aboutContent.problem.title}
                        </motion.h2>
                        <motion.div variants={fadeUp} className="flex items-baseline gap-4 mb-6">
                            <span className="font-display text-5xl text-[var(--primary)]">
                                {aboutContent.problem.stat}
                            </span>
                            <span className="text-sm text-[var(--fg-muted)] font-medium">
                                {aboutContent.problem.statLabel}
                            </span>
                        </motion.div>
                        <motion.p variants={fadeUp} className="text-[var(--fg-muted)] leading-relaxed max-w-2xl mb-8">
                            {aboutContent.problem.copy}
                        </motion.p>
                        <motion.div variants={fadeUp} className="overflow-x-auto">
                            <table className="w-full text-sm border-collapse">
                                <thead>
                                    <tr className="border-b border-[var(--border)]">
                                        <th className="text-left py-3 pr-4 font-semibold text-[var(--fg)]">Physician Age</th>
                                        <th className="text-left py-3 pr-4 font-semibold text-[var(--fg)]">Hearing Loss</th>
                                        <th className="text-left py-3 font-semibold text-[var(--fg)]">Clinical Impact</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {aboutContent.problem.table.map((row) => (
                                        <tr key={row.age} className="border-b border-[var(--border)]">
                                            <td className="py-3 pr-4 text-[var(--fg-muted)] font-mono">{row.age}</td>
                                            <td className="py-3 pr-4 text-[var(--fg-muted)]">{row.loss}</td>
                                            <td className="py-3 text-[var(--fg-muted)]">{row.impact}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </motion.div>
            </AnimatedSection>

            {/* Our Solution */}
            <AnimatedSection
                className="py-20 lg:py-28 bg-[var(--bg-elevated)] border-t border-[var(--border)]"
                contentClassName="max-w-4xl mx-auto px-6 lg:px-8"
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            02
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-6">
                            {aboutContent.solution.title}
                        </motion.h2>
                        <motion.p variants={fadeUp} className="text-xl font-semibold text-[var(--fg)] mb-6">
                            {aboutContent.solution.statement}
                        </motion.p>
                        <motion.p variants={fadeUp} className="text-sm font-semibold text-[var(--fg-muted)] mb-2">
                            Why a Progressive Web App
                        </motion.p>
                        <ul className="space-y-2 mb-8">
                            {aboutContent.solution.whyPwa.map((item, i) => (
                                <motion.li key={i} variants={fadeUp} className="flex items-start gap-3 text-[var(--fg-muted)] text-sm">
                                    <span className="w-1.5 h-1.5 rounded-full bg-[var(--primary)] mt-1.5 shrink-0" />
                                    {item}
                                </motion.li>
                            ))}
                        </ul>
                        <ul className="space-y-4">
                            {aboutContent.solution.bullets.map((bullet, i) => (
                                <motion.li key={i} variants={fadeUp} className="flex items-start gap-3 text-[var(--fg-muted)]">
                                    <span className="w-1.5 h-1.5 rounded-full bg-[var(--primary)] mt-2 shrink-0" />
                                    {bullet}
                                </motion.li>
                            ))}
                        </ul>
            </AnimatedSection>

            {/* Key Features */}
            <AnimatedSection
                className="py-20 lg:py-28 border-t border-[var(--border)]"
                contentClassName="max-w-6xl mx-auto px-6 lg:px-8"
                staggerChildren={0.05}
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            03
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-12">
                            Key Features
                        </motion.h2>
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                            {aboutContent.features.map((f) => {
                                const Icon = featureIcons[f.icon] ?? Activity
                                return (
                                    <motion.div key={f.label} variants={fadeUp} className="card-static p-5 flex flex-col items-center text-center gap-3">
                                        <div className="w-10 h-10 bg-[var(--primary)]/10 rounded-[var(--radius)] flex items-center justify-center">
                                            <Icon className="w-5 h-5 text-[var(--primary)]" strokeWidth={2} />
                                        </div>
                                        <span className="text-sm font-semibold text-[var(--fg)]">{f.label}</span>
                                    </motion.div>
                                )
                            })}
                        </div>
            </AnimatedSection>

            {/* Physician Workflow */}
            <AnimatedSection
                className="py-20 lg:py-28 bg-[var(--bg-elevated)] border-t border-[var(--border)]"
                contentClassName="max-w-4xl mx-auto px-6 lg:px-8"
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            04
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-12">
                            Physician Workflow
                        </motion.h2>
                        <div className="space-y-6">
                            {aboutContent.workflow.map((p) => (
                                <motion.div key={p.step} variants={fadeUp} className="flex items-start gap-4">
                                    <span className="font-display text-2xl font-normal text-[var(--primary)] shrink-0 w-8">
                                        {String(p.step).padStart(2, '0')}
                                    </span>
                                    <div>
                                        <div className="font-semibold text-[var(--fg)]">{p.label}</div>
                                        <div className="text-sm text-[var(--fg-muted)] mt-0.5">{p.desc}</div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
            </AnimatedSection>

            {/* Training Data */}
            <AnimatedSection
                className="py-20 lg:py-28 border-t border-[var(--border)]"
                contentClassName="max-w-4xl mx-auto px-6 lg:px-8"
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            05
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-6">
                            {aboutContent.trainingData.title}
                        </motion.h2>
                        <motion.p variants={fadeUp} className="text-[var(--fg-muted)] mb-4">
                            {aboutContent.trainingData.primary}
                        </motion.p>
                        <ul className="space-y-2">
                            {aboutContent.trainingData.supplementary.map((item, i) => (
                                <motion.li key={i} variants={fadeUp} className="flex items-start gap-3 text-sm text-[var(--fg-muted)]">
                                    <span className="w-1.5 h-1.5 rounded-full bg-[var(--primary)] mt-1.5 shrink-0" />
                                    {item}
                                </motion.li>
                            ))}
                        </ul>
            </AnimatedSection>

            {/* ML Pipeline */}
            <AnimatedSection
                className="py-20 lg:py-28 bg-[var(--bg-elevated)] border-t border-[var(--border)]"
                contentClassName="max-w-4xl mx-auto px-6 lg:px-8"
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            06
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-8">
                            Machine Learning Pipeline
                        </motion.h2>
                        <motion.div variants={fadeUp} className="space-y-6">
                            <div>
                                <div className="text-sm font-semibold text-[var(--fg)] mb-1">Feature Extraction</div>
                                <p className="text-sm text-[var(--fg-muted)]">{aboutContent.mlPipeline.features}</p>
                            </div>
                            <div>
                                <div className="text-sm font-semibold text-[var(--fg)] mb-1">Model</div>
                                <p className="text-sm text-[var(--fg-muted)]">{aboutContent.mlPipeline.model}</p>
                            </div>
                            <div>
                                <div className="text-sm font-semibold text-[var(--fg)] mb-1">Explainability</div>
                                <p className="text-sm text-[var(--fg-muted)]">{aboutContent.mlPipeline.explainability}</p>
                            </div>
                        </motion.div>
            </AnimatedSection>

            {/* Technology */}
            <AnimatedSection
                className="py-20 lg:py-28 border-t border-[var(--border)]"
                contentClassName="max-w-4xl mx-auto px-6 lg:px-8"
                staggerChildren={0.03}
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            07
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-8">
                            Technology
                        </motion.h2>
                        <div className="space-y-6">
                            <div>
                                <div className="text-xs font-semibold text-[var(--fg-muted)] uppercase tracking-wider mb-2">Frontend</div>
                                <div className="flex flex-wrap gap-2">
                                    {aboutContent.techStack.frontend.map((tech) => (
                                        <motion.span key={tech} variants={fadeUp} className="px-3 py-1.5 bg-[var(--bg-muted)] text-[var(--fg)] font-mono text-xs font-medium rounded-[var(--radius)]">
                                            {tech}
                                        </motion.span>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs font-semibold text-[var(--fg-muted)] uppercase tracking-wider mb-2">Backend & ML</div>
                                <div className="flex flex-wrap gap-2">
                                    {aboutContent.techStack.backend.map((tech) => (
                                        <motion.span key={tech} variants={fadeUp} className="px-3 py-1.5 bg-[var(--bg-muted)] text-[var(--fg)] font-mono text-xs font-medium rounded-[var(--radius)]">
                                            {tech}
                                        </motion.span>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs font-semibold text-[var(--fg-muted)] uppercase tracking-wider mb-2">Infrastructure</div>
                                <div className="flex flex-wrap gap-2">
                                    {aboutContent.techStack.infra.map((tech) => (
                                        <motion.span key={tech} variants={fadeUp} className="px-3 py-1.5 bg-[var(--bg-muted)] text-[var(--fg)] font-mono text-xs font-medium rounded-[var(--radius)]">
                                            {tech}
                                        </motion.span>
                                    ))}
                                </div>
                            </div>
                        </div>
            </AnimatedSection>

            {/* Impact */}
            <AnimatedSection
                className="py-20 lg:py-28 bg-[var(--bg-elevated)] border-t border-[var(--border)]"
                contentClassName="max-w-6xl mx-auto px-6 lg:px-8"
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            08
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-12">
                            Impact & Use Cases
                        </motion.h2>
                        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
                            {aboutContent.impact.map((item) => (
                                <motion.div key={item.title} variants={fadeUp} className="card-static p-6">
                                    <h3 className="font-semibold text-[var(--fg)] mb-2">{item.title}</h3>
                                    <p className="text-sm text-[var(--fg-muted)] leading-relaxed">{item.desc}</p>
                                </motion.div>
                            ))}
                        </div>
            </AnimatedSection>

            {/* Roadmap */}
            <AnimatedSection
                className="py-20 lg:py-28 border-t border-[var(--border)]"
                contentClassName="max-w-4xl mx-auto px-6 lg:px-8"
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            09
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-8">
                            Product Roadmap
                        </motion.h2>
                        <div className="space-y-6">
                            <motion.div variants={fadeUp}>
                                <div className="text-sm font-semibold text-[var(--primary)] mb-1">Phase 1 — Hackathon MVP</div>
                                <p className="text-sm text-[var(--fg-muted)]">{aboutContent.roadmap.phase1}</p>
                            </motion.div>
                            <motion.div variants={fadeUp}>
                                <div className="text-sm font-semibold text-[var(--fg)] mb-1">Phase 2 — Clinical Beta (Months 1–3)</div>
                                <p className="text-sm text-[var(--fg-muted)]">{aboutContent.roadmap.phase2}</p>
                            </motion.div>
                            <motion.div variants={fadeUp}>
                                <div className="text-sm font-semibold text-[var(--fg)] mb-1">Phase 3 — FDA Pathway (Months 4–12)</div>
                                <p className="text-sm text-[var(--fg-muted)]">{aboutContent.roadmap.phase3}</p>
                            </motion.div>
                        </div>
            </AnimatedSection>

            {/* Demo Script */}
            <AnimatedSection
                className="py-20 lg:py-28 bg-[var(--bg-elevated)] border-t border-[var(--border)]"
                contentClassName="max-w-4xl mx-auto px-6 lg:px-8"
                staggerChildren={0.05}
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            10
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-8">
                            {aboutContent.demo.tagline}
                        </motion.h2>
                        <div className="space-y-4">
                            {aboutContent.demo.points.map((point, i) => (
                                <motion.p key={i} variants={fadeUp} className="text-sm text-[var(--fg-muted)] pl-4 border-l-2 border-[var(--border)]">
                                    {point}
                                </motion.p>
                            ))}
                        </div>
            </AnimatedSection>

            {/* Team */}
            <AnimatedSection
                className="py-20 lg:py-28 border-t border-[var(--border)]"
                contentClassName="max-w-4xl mx-auto px-6 lg:px-8"
            >
                <motion.span variants={fadeUp} className="text-xs font-semibold text-[var(--primary)] uppercase tracking-[0.2em]">
                            11
                        </motion.span>
                        <motion.h2 variants={fadeUp} className="text-2xl font-bold text-[var(--fg)] mt-2 mb-12">
                            Team
                        </motion.h2>
                        <div className="space-y-6 max-w-xl">
                            {aboutContent.team.map((member) => (
                                <motion.div key={member.name} variants={fadeUp} className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-[var(--radius)] bg-[var(--primary)]/10 flex items-center justify-center shrink-0">
                                        <span className="text-sm font-bold text-[var(--primary)]">{member.name.charAt(0)}</span>
                                    </div>
                                    <div>
                                        <div className="font-semibold text-[var(--fg)]">{member.name}</div>
                                        <div className="text-sm text-[var(--fg-muted)]">{member.role}</div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
            </AnimatedSection>

            {/* CTA */}
            <section className="py-20 lg:py-28 border-t border-[var(--border)]">
                <div className="max-w-4xl mx-auto px-6 lg:px-8 text-center">
                    <motion.p
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                        className="text-lg text-[var(--fg-muted)] mb-8"
                    >
                        Ready to see it in action?
                    </motion.p>
                    <Link
                        to="/record"
                        className="inline-flex items-center gap-2 px-8 py-4 bg-[var(--primary)] text-white font-semibold rounded-[var(--radius)] hover:bg-[var(--primary-hover)] transition-colors duration-150 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]"
                    >
                        Try RespiLens
                        <ArrowRight className="w-5 h-5" strokeWidth={2} />
                    </Link>
                    <p className="mt-8 text-xs text-[var(--fg-subtle)] max-w-md mx-auto leading-relaxed">
                        RespiLens is an AI-assisted decision support tool. It does not constitute medical advice and should not replace professional clinical judgment.
                    </p>
                </div>
            </section>
        </div>
    )
}
