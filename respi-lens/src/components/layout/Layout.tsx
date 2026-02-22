import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { LayoutDashboard, Mic, History, Info, Menu, X } from 'lucide-react'
import { Logo } from './Logo'
import { useState } from 'react'

const navLinks = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/record', label: 'Record', icon: Mic },
    { to: '/history', label: 'History', icon: History },
    { to: '/about', label: 'About', icon: Info },
]

const pageTransition = {
    initial: { opacity: 0, y: 8 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -4 },
    transition: { duration: 0.2 },
}

export function Layout() {
    const [mobileOpen, setMobileOpen] = useState(false)
    const location = useLocation()

    return (
        <div className="min-h-screen bg-[var(--bg)] text-[var(--fg)] flex flex-col">
            <header className="sticky top-0 z-50 bg-[var(--bg-elevated)]/95 backdrop-blur-md border-b border-[var(--border)] shadow-sm">
                <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="h-14 flex items-center justify-between md:grid md:grid-cols-[1fr_auto_1fr] gap-6">
                        <div className="flex items-center min-w-0">
                            <Logo />
                        </div>

                        {/* Centered nav — grid keeps it visually centered */}
                        <nav className="hidden md:flex items-center gap-1 justify-self-center">
                            {navLinks.map(({ to, label, icon: Icon }) => (
                                <NavLink
                                    key={to}
                                    to={to}
                                    end={to === '/'}
                                    className={({ isActive }) =>
                                        `flex items-center gap-1.5 px-3.5 py-1.5 text-[13px] font-semibold rounded-full transition-all duration-150 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]
                                        ${isActive
                                            ? 'bg-[var(--primary)] text-white shadow-sm'
                                            : 'text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bg-[var(--bg-muted)]'
                                        }`
                                    }
                                >
                                    <Icon className="w-3.5 h-3.5" strokeWidth={2} />
                                    {label}
                                </NavLink>
                            ))}
                        </nav>

                        {/* Right cell: menu button on mobile, empty on desktop for balance */}
                        <div className="flex items-center justify-end">
                            <button
                                onClick={() => setMobileOpen(!mobileOpen)}
                                className="md:hidden p-2 rounded-full text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bg-[var(--bg-muted)] transition-colors duration-150 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]"
                                aria-label="Toggle navigation"
                            >
                                {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                            </button>
                        </div>
                    </div>
                </div>

                <AnimatePresence>
                    {mobileOpen && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.2 }}
                            className="md:hidden border-t border-[var(--border)] bg-[var(--bg-elevated)]"
                        >
                            <nav className="flex flex-col p-4 gap-1">
                                {navLinks.map(({ to, label, icon: Icon }) => (
                                    <NavLink
                                        key={to}
                                        to={to}
                                        end={to === '/'}
                                        onClick={() => setMobileOpen(false)}
                                        className={({ isActive }) =>
                                            `flex items-center gap-3 px-4 py-3 rounded-[var(--radius)] text-sm font-medium transition-colors duration-150
                                            ${isActive
                                                ? 'bg-[var(--primary)] text-white'
                                                : 'text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bg-[var(--bg-muted)]'
                                            }`
                                        }
                                    >
                                        <Icon className="w-4 h-4" />
                                        {label}
                                    </NavLink>
                                ))}
                            </nav>
                        </motion.div>
                    )}
                </AnimatePresence>
            </header>

            <main className="flex-1">
                <AnimatePresence mode="wait" initial={false}>
                    <motion.div
                        key={location.pathname}
                        initial={pageTransition.initial}
                        animate={pageTransition.animate}
                        exit={pageTransition.exit}
                        transition={pageTransition.transition}
                    >
                        <Outlet />
                    </motion.div>
                </AnimatePresence>
            </main>

            <footer className="border-t-2 border-[var(--primary)]/15 px-6 py-5">
                <div className="max-w-6xl mx-auto space-y-3">
                    <nav className="flex flex-wrap items-center justify-center gap-x-6 gap-y-1 text-xs">
                        {navLinks.map(({ to, label }) => (
                            <NavLink
                                key={to}
                                to={to}
                                end={to === '/'}
                                className="text-[var(--fg-subtle)] hover:text-[var(--primary)] transition-colors duration-150"
                            >
                                {label}
                            </NavLink>
                        ))}
                    </nav>
                    <div className="flex items-center justify-between text-xs text-[var(--fg-subtle)]">
                        <span>Respi Clinical — AI-Powered Lung Sound Analysis</span>
                        <span className="font-mono">v0.1.0 · PWA</span>
                    </div>
                </div>
            </footer>
        </div>
    )
}
