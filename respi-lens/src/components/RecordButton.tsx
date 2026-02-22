import { Mic, Square } from 'lucide-react'
import { useRecordingStore } from '../stores/recordingStore'
import { useAudioRecorder } from '../hooks/useAudioRecorder'
import { motion } from 'framer-motion'

export function RecordButton() {
    const { isRecording } = useRecordingStore()
    const { startRecording, stopRecording } = useAudioRecorder()

    return (
        <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={isRecording ? stopRecording : startRecording}
            className={`flex items-center gap-2 px-6 py-3 rounded-[var(--radius)] font-semibold transition-colors duration-150 ${
                isRecording
                    ? 'bg-[var(--destructive)] hover:opacity-90 text-white'
                    : 'bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white'
            }`}
        >
            {isRecording ? (
                <>
                    <Square className="w-5 h-5" strokeWidth={2} />
                    Stop Recording
                </>
            ) : (
                <>
                    <Mic className="w-5 h-5" strokeWidth={2} />
                    Start Recording
                </>
            )}
        </motion.button>
    )
}
