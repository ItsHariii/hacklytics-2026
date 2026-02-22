import { useRef, useState, useEffect, useCallback, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, Volume2 } from 'lucide-react'
import type { DetectedSegment } from '../types'
import { CLASSIFICATION_CONFIG } from '../lib/theme'
import type { ClassificationKey } from '../lib/theme'

function getCSSVar(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

interface AudioEventPlayerProps {
    audioUrl: string | undefined
    durationSeconds: number
    segments: DetectedSegment[]
    classification: string
}

function getSegmentConfig(soundType: string) {
    const key = soundType as ClassificationKey
    const config = CLASSIFICATION_CONFIG[key] ?? CLASSIFICATION_CONFIG.crackles
    return {
        bg: config.segmentBg,
        border: config.segmentBorder,
        chip: config.segmentChip,
        text: config.label,
        hex: config.hexColor,
        shortLabel: soundType === 'crackles' ? 'C' : soundType === 'wheezes' ? 'W' : 'B',
    }
}

function formatTime(sec: number): string {
    const m = Math.floor(sec / 60)
    const s = Math.floor(sec % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
}

function buildEventSummary(segments: DetectedSegment[], totalDuration: number): string {
    const crackleCount = segments.filter((s) => s.sound_type === 'crackles').length
    const wheezeCount = segments.filter((s) => s.sound_type === 'wheezes').length
    const bothCount = segments.filter((s) => s.sound_type === 'both').length
    const parts: string[] = []
    if (crackleCount > 0) parts.push(`${crackleCount} crackle event${crackleCount > 1 ? 's' : ''}`)
    if (wheezeCount > 0) parts.push(`${wheezeCount} wheeze event${wheezeCount > 1 ? 's' : ''}`)
    if (bothCount > 0) parts.push(`${bothCount} mixed event${bothCount > 1 ? 's' : ''}`)
    if (parts.length === 0) return 'No adventitious sounds detected'
    return `${parts.join(' and ')} detected across ${totalDuration.toFixed(1)}s recording`
}

export function AudioEventPlayer({
    audioUrl,
    durationSeconds,
    segments,
    classification,
}: AudioEventPlayerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const audioCtxRef = useRef<AudioContext | null>(null)
    const sourceRef = useRef<AudioBufferSourceNode | null>(null)
    const audioBufferRef = useRef<AudioBuffer | null>(null)
    const animFrameRef = useRef<number>(0)

    const [waveformData, setWaveformData] = useState<Float32Array | null>(null)
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrent] = useState(0)
    const [totalDuration, setTotalDuration] = useState(durationSeconds)
    const [activeSegment, setActiveSegment] = useState<number | null>(null)
    const [loadError, setLoadError] = useState(false)

    const startTimeRef = useRef(0)
    const offsetRef = useRef(0)

    useEffect(() => {
        if (!audioUrl || audioUrl.trim() === '') {
            setLoadError(true)
            return
        }

        let cancelled = false
        const ctx = new AudioContext()
        audioCtxRef.current = ctx

        fetch(audioUrl)
            .then((res) => {
                if (!res.ok) throw new Error('fetch failed')
                return res.arrayBuffer()
            })
            .then((buf) => ctx.decodeAudioData(buf))
            .then((decoded) => {
                if (cancelled) return
                audioBufferRef.current = decoded
                setTotalDuration(decoded.duration)
                const channel = decoded.getChannelData(0)
                const samples = 600
                const blockSize = Math.floor(channel.length / samples)
                const peaks = new Float32Array(samples)
                for (let i = 0; i < samples; i++) {
                    let sum = 0
                    for (let j = 0; j < blockSize; j++) {
                        sum += Math.abs(channel[i * blockSize + j])
                    }
                    peaks[i] = sum / blockSize
                }
                setWaveformData(peaks)
            })
            .catch(() => {
                if (!cancelled) setLoadError(true)
            })

        return () => {
            cancelled = true
            ctx.close()
        }
    }, [audioUrl])

    useEffect(() => {
        if (audioUrl && audioUrl.trim() !== '') {
            setLoadError(false)
        }
    }, [audioUrl])

    const draw = useCallback(() => {
        const canvas = canvasRef.current
        if (!canvas || !waveformData) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const dpr = window.devicePixelRatio || 1
        const rect = canvas.getBoundingClientRect()
        canvas.width = rect.width * dpr
        canvas.height = rect.height * dpr
        ctx.scale(dpr, dpr)

        const W = rect.width
        const H = rect.height
        const dur = totalDuration || durationSeconds

        ctx.clearRect(0, 0, W, H)
        ctx.fillStyle = getCSSVar('--bg-muted')
        ctx.fillRect(0, 0, W, H)

        for (const seg of segments) {
            const x0 = (seg.start_sec / dur) * W
            const x1 = (seg.end_sec / dur) * W
            const colors = getSegmentConfig(seg.sound_type)
            ctx.fillStyle = colors.bg
            ctx.fillRect(x0, 0, x1 - x0, H)
            ctx.strokeStyle = colors.border
            ctx.lineWidth = 1
            ctx.strokeRect(x0, 0, x1 - x0, H)

            // Draw segment label directly on waveform
            const segW = x1 - x0
            if (segW > 30) {
                ctx.font = '600 10px system-ui, sans-serif'
                ctx.fillStyle = colors.hex
                ctx.globalAlpha = 0.8
                ctx.textAlign = 'center'
                ctx.textBaseline = 'top'
                ctx.fillText(colors.text, x0 + segW / 2, 4)
                ctx.globalAlpha = 1.0
            }
        }

        const maxVal = Math.max(...waveformData) || 1
        const barW = W / waveformData.length
        for (let i = 0; i < waveformData.length; i++) {
            const amplitude = (waveformData[i] / maxVal) * (H * 0.8)
            const x = i * barW
            const y = (H - amplitude) / 2

            const t = i / waveformData.length
            const playheadT = currentTime / dur
            ctx.fillStyle = t <= playheadT ? getCSSVar('--primary') : getCSSVar('--fg-subtle')
            ctx.fillRect(x, y, Math.max(barW - 0.5, 1), amplitude)
        }

        const px = (currentTime / dur) * W
        ctx.strokeStyle = getCSSVar('--primary')
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(px, 0)
        ctx.lineTo(px, H)
        ctx.stroke()
    }, [waveformData, segments, currentTime, totalDuration, durationSeconds])

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

    useEffect(() => {
        if (!isPlaying) return

        const tick = () => {
            const ctx = audioCtxRef.current
            if (!ctx) return
            const elapsed = ctx.currentTime - startTimeRef.current + offsetRef.current
            const dur = totalDuration || durationSeconds
            if (elapsed >= dur) {
                stopPlayback()
                return
            }
            setCurrent(elapsed)
            animFrameRef.current = requestAnimationFrame(tick)
        }
        animFrameRef.current = requestAnimationFrame(tick)

        return () => cancelAnimationFrame(animFrameRef.current)
    }, [isPlaying, totalDuration, durationSeconds])

    const stopPlayback = useCallback(() => {
        try {
            sourceRef.current?.stop()
        } catch { /* already stopped */ }
        sourceRef.current = null
        setIsPlaying(false)
        setActiveSegment(null)
        cancelAnimationFrame(animFrameRef.current)
    }, [])

    const playRange = useCallback(
        (startSec: number, endSec?: number) => {
            const ctx = audioCtxRef.current
            const buf = audioBufferRef.current
            if (!ctx || !buf) return

            stopPlayback()

            const source = ctx.createBufferSource()
            source.buffer = buf
            source.connect(ctx.destination)

            const duration = endSec ? endSec - startSec : undefined
            source.start(0, startSec, duration)
            source.onended = () => {
                setIsPlaying(false)
                setActiveSegment(null)
            }

            sourceRef.current = source
            startTimeRef.current = ctx.currentTime
            offsetRef.current = startSec
            setCurrent(startSec)
            setIsPlaying(true)
        },
        [stopPlayback],
    )

    const togglePlay = useCallback(() => {
        if (isPlaying) {
            stopPlayback()
        } else {
            playRange(0)
        }
    }, [isPlaying, stopPlayback, playRange])

    const playSegment = useCallback(
        (idx: number) => {
            const seg = segments[idx]
            if (!seg || typeof seg.start_sec !== 'number' || typeof seg.end_sec !== 'number') return
            setActiveSegment(idx)
            playRange(seg.start_sec, seg.end_sec)
        },
        [segments, playRange],
    )

    const handleCanvasClick = useCallback(
        (e: React.MouseEvent<HTMLCanvasElement>) => {
            const canvas = canvasRef.current
            if (!canvas) return
            const rect = canvas.getBoundingClientRect()
            const x = e.clientX - rect.left
            const t = (x / rect.width) * (totalDuration || durationSeconds)

            const hitIdx = segments.findIndex(
                (s) => t >= s.start_sec && t <= s.end_sec,
            )
            if (hitIdx !== -1) {
                playSegment(hitIdx)
            } else {
                playRange(t)
            }
        },
        [segments, totalDuration, durationSeconds, playSegment, playRange],
    )

    const eventSummary = useMemo(
        () => buildEventSummary(segments, totalDuration),
        [segments, totalDuration],
    )

    if (loadError) {
        return (
            <div className="card-static p-6 mb-6">
                <div className="flex items-center gap-3 text-[var(--fg-muted)]">
                    <Volume2 className="w-5 h-5" strokeWidth={2} />
                    <span className="text-sm font-medium">Audio playback unavailable for this recording</span>
                </div>
            </div>
        )
    }

    const hasSegments = Array.isArray(segments) && segments.length > 0
    const dur = totalDuration || durationSeconds

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="card-static p-6 mb-6"
        >
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Volume2 className="w-4 h-4 text-[var(--fg-muted)]" strokeWidth={2} />
                    <span className="text-sm font-semibold text-[var(--fg)]">Lung Sound Recording</span>
                </div>
                <span className="text-xs font-mono font-medium text-[var(--fg-muted)]">
                    {formatTime(currentTime)} / {formatTime(totalDuration)}
                </span>
            </div>

            {/* Waveform Canvas */}
            <div ref={containerRef} className="relative mb-2">
                <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    className="w-full h-24 rounded-[var(--radius)] cursor-pointer block"
                    style={{ display: 'block' }}
                />
                {!waveformData && !loadError && (
                    <div className="absolute inset-0 flex items-center justify-center rounded-[var(--radius)] bg-[var(--bg-muted)]">
                        <div className="w-5 h-5 border-2 border-[var(--primary)] border-t-transparent rounded-full animate-spin" />
                    </div>
                )}
            </div>

            {/* Segment Timeline Strip */}
            {hasSegments && (
                <div className="relative h-6 bg-[var(--bg-muted)] rounded-[var(--radius-sm)] mb-4 overflow-hidden">
                    {segments.map((seg, idx) => {
                        if (typeof seg?.start_sec !== 'number' || typeof seg?.end_sec !== 'number') return null
                        const colors = getSegmentConfig(seg.sound_type)
                        const left = (seg.start_sec / dur) * 100
                        const width = ((seg.end_sec - seg.start_sec) / dur) * 100
                        return (
                            <button
                                key={`tl-${idx}`}
                                onClick={() => playSegment(idx)}
                                className="absolute top-0 h-full flex items-center justify-center text-[9px] font-bold cursor-pointer transition-opacity hover:opacity-90"
                                style={{
                                    left: `${left}%`,
                                    width: `${Math.max(width, 1.5)}%`,
                                    backgroundColor: colors.bg,
                                    borderLeft: `1px solid ${colors.border}`,
                                    borderRight: `1px solid ${colors.border}`,
                                    color: colors.hex,
                                }}
                                title={`${colors.text}: ${seg.start_sec.toFixed(1)}s – ${seg.end_sec.toFixed(1)}s`}
                            >
                                {width > 3 ? colors.shortLabel : ''}
                            </button>
                        )
                    })}
                </div>
            )}

            {/* Play controls + summary */}
            <div className="flex items-center gap-4 mb-4">
                <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={togglePlay}
                    disabled={!waveformData}
                    className="w-10 h-10 rounded-[var(--radius)] bg-[var(--primary)] text-white flex items-center justify-center disabled:bg-[var(--bg-muted)] disabled:cursor-not-allowed transition-colors duration-150"
                >
                    {isPlaying ? (
                        <Pause className="w-5 h-5" strokeWidth={2} />
                    ) : (
                        <Play className="w-5 h-5 ml-0.5" strokeWidth={2} />
                    )}
                </motion.button>
                <span className="text-xs text-[var(--fg-muted)]">
                    {hasSegments
                        ? eventSummary
                        : classification === 'normal'
                            ? 'No adventitious sounds detected'
                            : 'Click waveform to play from any point'}
                </span>
            </div>

            {/* Event Table */}
            {hasSegments && (
                <div className="border border-[var(--border)] rounded-[var(--radius)] overflow-hidden">
                    <div className="grid grid-cols-[2rem_1fr_auto_auto_auto_2.5rem] gap-x-3 px-3 py-2 bg-[var(--bg-muted)] text-[10px] font-semibold text-[var(--fg-muted)] uppercase tracking-wide">
                        <span>#</span>
                        <span>Type</span>
                        <span>Time Range</span>
                        <span>Duration</span>
                        <span>Conf.</span>
                        <span />
                    </div>
                    {segments.map((seg, idx) => {
                        if (typeof seg?.start_sec !== 'number' || typeof seg?.end_sec !== 'number') return null
                        const colors = getSegmentConfig(seg.sound_type)
                        const isActive = activeSegment === idx && isPlaying
                        const segDuration = seg.end_sec - seg.start_sec
                        return (
                            <div
                                key={`ev-${idx}`}
                                className={`grid grid-cols-[2rem_1fr_auto_auto_auto_2.5rem] gap-x-3 px-3 py-2 items-center text-xs border-t border-[var(--border)] transition-colors duration-100 ${
                                    isActive ? 'bg-[var(--bg-muted)]' : 'hover:bg-[var(--bg-muted)]/50'
                                }`}
                            >
                                <span className="text-[var(--fg-muted)] font-mono">{idx + 1}</span>
                                <div className="flex items-center gap-2">
                                    <span
                                        className="w-2 h-2 rounded-full shrink-0"
                                        style={{ backgroundColor: colors.hex }}
                                    />
                                    <span className="font-medium text-[var(--fg)]">{colors.text}</span>
                                </div>
                                <span className="font-mono text-[var(--fg-muted)]">
                                    {seg.start_sec.toFixed(1)}s – {seg.end_sec.toFixed(1)}s
                                </span>
                                <span className="font-mono text-[var(--fg-muted)]">
                                    {segDuration.toFixed(1)}s
                                </span>
                                <span className="font-semibold" style={{ color: colors.hex }}>
                                    {Math.round(seg.confidence * 100)}%
                                </span>
                                <button
                                    onClick={() => playSegment(idx)}
                                    className="w-6 h-6 rounded-[var(--radius-sm)] bg-[var(--bg-muted)] flex items-center justify-center hover:bg-[var(--border)] transition-colors duration-100"
                                >
                                    {isActive ? (
                                        <Pause className="w-3 h-3 text-[var(--fg)]" strokeWidth={2} />
                                    ) : (
                                        <Play className="w-3 h-3 text-[var(--fg)] ml-px" strokeWidth={2} />
                                    )}
                                </button>
                            </div>
                        )
                    })}
                </div>
            )}
        </motion.div>
    )
}
