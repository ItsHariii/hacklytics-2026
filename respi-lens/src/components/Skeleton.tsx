import { motion } from 'framer-motion'

interface SkeletonProps {
    className?: string
    animate?: boolean
}

export function Skeleton({ className = '', animate = true }: SkeletonProps) {
    return (
        <motion.div
            className={`bg-[var(--bg-muted)] rounded-[var(--radius-sm)] ${className}`}
            animate={animate ? { opacity: [0.5, 0.8, 0.5] } : {}}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
        />
    )
}
