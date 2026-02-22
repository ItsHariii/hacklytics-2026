import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

interface AnimatedSectionProps extends React.HTMLAttributes<HTMLElement> {
    children: React.ReactNode
    className?: string
    /** Inner container class (e.g. max-w-4xl mx-auto px-6 lg:px-8). Required for stagger to work with direct motion children. */
    contentClassName?: string
    /** useInView margin (e.g. '-80px'). See framer-motion docs. */
    margin?: string
    staggerChildren?: number
}

export function AnimatedSection({
    children,
    className,
    contentClassName,
    margin = '-80px',
    staggerChildren = 0.06,
    ...props
}: AnimatedSectionProps) {
    const ref = useRef<HTMLElement>(null)
    const inView = useInView(ref, {
        once: true,
        ...(margin && { margin: margin as `${number}px` }),
    })

    return (
        <section ref={ref} className={className} {...props}>
            {contentClassName ? (
                <div className={contentClassName}>
                    <motion.div
                        initial="hidden"
                        animate={inView ? 'visible' : 'hidden'}
                        variants={{
                            hidden: {},
                            visible: { transition: { staggerChildren } },
                        }}
                    >
                        {children}
                    </motion.div>
                </div>
            ) : (
                <motion.div
                    initial="hidden"
                    animate={inView ? 'visible' : 'hidden'}
                    variants={{
                        hidden: {},
                        visible: { transition: { staggerChildren } },
                    }}
                >
                    {children}
                </motion.div>
            )}
        </section>
    )
}
