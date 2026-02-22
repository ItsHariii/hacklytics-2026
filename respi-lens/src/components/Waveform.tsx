import { useEffect, useRef, useCallback } from 'react'
import { useRecordingStore } from '../stores/recordingStore'

function getCSSVar(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export function Waveform() {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const { waveformData, isRecording } = useRecordingStore()

    const draw = useCallback(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const dpr = window.devicePixelRatio || 1
        const rect = canvas.getBoundingClientRect()
        canvas.width = rect.width * dpr
        canvas.height = rect.height * dpr
        ctx.scale(dpr, dpr)

        const W = rect.width
        const H = rect.height
        const centerY = H / 2

        ctx.fillStyle = getCSSVar('--bg-muted')
        ctx.fillRect(0, 0, W, H)

        if (waveformData.length === 0) return

        ctx.fillStyle = isRecording ? getCSSVar('--destructive') : getCSSVar('--primary')

        const barCount = Math.min(150, Math.floor(W / 3))
        const barWidth = Math.max(1, (W / barCount) * 0.7)
        const gap = (W / barCount) - barWidth

        for (let i = 0; i < barCount; i++) {
            const idx = Math.floor(i * (waveformData.length / barCount))
            const v = waveformData[idx] ?? 128
            const normalized = (v - 128) / 128
            const barHeight = Math.abs(normalized) * (H * 0.45)
            const x = i * (W / barCount) + gap / 2

            if (normalized >= 0) {
                ctx.fillRect(x, centerY - barHeight, barWidth, barHeight)
            } else {
                ctx.fillRect(x, centerY, barWidth, barHeight)
            }
        }
    }, [waveformData, isRecording])

    useEffect(() => {
        draw()
    }, [draw])

    useEffect(() => {
        const container = containerRef.current
        if (!container) return
        const ro = new ResizeObserver(() => draw())
        ro.observe(container)
        return () => ro.disconnect()
    }, [draw])

    return (
        <div ref={containerRef} className="w-full">
            <canvas
                ref={canvasRef}
                className="w-full h-[150px] rounded-[var(--radius)] border border-[var(--border)] block"
            />
        </div>
    )
}
