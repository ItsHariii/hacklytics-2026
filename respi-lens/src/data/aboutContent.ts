export const aboutContent = {
    hero: {
        tagline: 'Accessible via URL. Clinical-grade quality. No compromise.',
        headline: 'Built for clinicians. Powered by deep learning.',
        subheadline:
            '25% of primary care physicians in the US are over 60. Age-related hearing loss disproportionately affects the 2–5 kHz range — exactly where fine crackles occur. RespiLens applies deep learning to lung sound recordings, classifying adventitious sounds with SHAP-based explainability so clinicians understand the acoustic evidence behind every result. As a Progressive Web App, any physician can use it today by opening a URL — no installation, no app store, no IT approval.',
    },
    problem: {
        title: 'The Problem',
        stat: '25%',
        statLabel: 'of US primary care physicians are over 60',
        copy: 'Auscultation remains the cornerstone of respiratory diagnosis, yet the human ear is a biological instrument with measurable limitations that worsen with age. Fine crackles — the earliest indicator of pulmonary fibrosis, interstitial lung disease, and early-stage pneumonia — occur at 2–5 kHz, the range presbycusis degrades first. A 65-year-old physician may genuinely miss bilateral crackles that a younger clinician would catch. Meanwhile, medical software distribution is broken: hospital MDM policies block app downloads, IT approval takes weeks, and patients wait. A web application eliminates every barrier.',
        table: [
            { age: '40–50 years', loss: 'Subtle loss > 4 kHz', impact: 'May miss fine crackles' },
            { age: '50–60 years', loss: 'Moderate loss > 3 kHz', impact: 'Crackles harder to detect' },
            { age: '60+ years', loss: 'Significant loss > 2 kHz', impact: 'May rely on wheeze only' },
        ],
    },
    solution: {
        title: 'Our Solution',
        statement: 'RespiLens Clinical — a Progressive Web App that gives stethoscopes superhuman hearing.',
        whyPwa: [
            'Instant deployment — physicians access via URL, no installation required',
            'Universal compatibility — works on desktop, tablet, and mobile with a single codebase',
            'Clinical accessibility — bypasses hospital IT restrictions and app store delays',
        ],
        bullets: [
            'Classifies adventitious sounds: crackles vs wheezes, fine vs coarse, monophonic vs polyphonic',
            'Identifies timing in the respiratory cycle: inspiratory vs expiratory phase',
            'SHAP explainability — clinicians see exactly which acoustic features influenced the result',
            'Web Audio API for clinical-grade signal processing at 44.1 kHz, entirely in the browser',
        ],
    },
    workflow: [
        { step: 1, label: 'Open URL', desc: 'Navigate to RespiLens in any modern browser. Instant access — no login required for demo mode.' },
        { step: 2, label: 'Connect Input', desc: 'Pair a Bluetooth digital stethoscope via Web Bluetooth, connect a wired mic, or upload a .wav file.' },
        { step: 3, label: 'Record', desc: 'Auscultate for 10–30 seconds. Web Audio API captures at 44.1 kHz with a live waveform display.' },
        { step: 4, label: 'Analyze', desc: 'Respiratory cycles are extracted, features computed, and a multi-task CNN runs inference with SHAP attribution.' },
        { step: 5, label: 'Review Results', desc: 'Sound classification, disease probabilities, severity score, and SHAP explanation delivered in real time.' },
        { step: 6, label: 'Clinical Decision', desc: 'Review AI findings alongside your own assessment. Download a PDF report or save to patient history.' },
    ],
    features: [
        { label: 'Web Audio capture', icon: 'Mic' },
        { label: 'Live waveform', icon: 'Activity' },
        { label: '4-class classification', icon: 'TrendingUp' },
        { label: 'SHAP explainability', icon: 'BarChart' },
        { label: '7-day trend chart', icon: 'LineChart' },
        { label: 'PDF clinical report', icon: 'FileText' },
    ],
    trainingData: {
        title: 'Training Data',
        primary: 'ICBHI 2017 Respiratory Sound Database — 6,898 annotated respiratory cycles from 126 patients across 5.5 hours of recordings, labeled by respiratory medicine experts.',
        supplementary: [
            'PKU Respiratory Dataset — 11,968 cycles including COVID-19 pneumonia cases',
            'HF_Lung_V2 — Extended labels for inspiratory vs expiratory phase',
            'Mendeley Lung Sounds — 112 patients across multiple chest positions and filter settings',
        ],
    },
    mlPipeline: {
        features: 'MFCCs (20 coefficients), mel-spectrogram (128 bins, 50–2000 Hz), zero-crossing rate, RMS energy, harmonic-to-noise ratio, and spectral flux.',
        model: 'Multi-task CNN with dual input branches (spectrogram + tabular features) and three output heads: sound classification, disease probability, and severity score.',
        explainability: 'Every classification includes SHAP values highlighting the top 5 contributing features with plain-language labels physicians can interpret.',
    },
    techStack: {
        frontend: ['React 18', 'Vite', 'TypeScript', 'Tailwind CSS', 'Recharts', 'Framer Motion', 'Zustand', 'PWA'],
        backend: ['Python 3.11', 'FastAPI', 'PyTorch', 'librosa', 'SHAP', 'ONNX Runtime'],
        infra: ['Supabase', 'Vercel', 'Railway', 'Web Audio API', 'Web Bluetooth API'],
    },
    impact: [
        { title: 'Physician hearing support', desc: 'Compensate for age-related hearing loss — detect fine crackles at 2–5 kHz that presbycusis may obscure.' },
        { title: 'Zero-friction deployment', desc: 'No IT ticket, no MDM enrollment, no app store. Open a URL and start using it in clinic today.' },
        { title: 'Longitudinal monitoring', desc: '7-day severity trend charts surface deteriorating patients at a glance — patterns a single visit might miss.' },
        { title: 'Clinical documentation', desc: 'Generate PDF reports in seconds for EHR integration, specialist referrals, or patient records.' },
    ],
    roadmap: {
        phase1: 'Hackathon MVP — Web Audio capture, 4-class lung sound classification, SHAP explainability, real-time results via Supabase Realtime, severity trends, PDF reports, and full PWA support.',
        phase2: 'Native mobile apps (React Native), multi-dataset training with ICBHI + PKU for improved accuracy, EHR integration pilot via FHIR, teaching mode for medical students, and clinic-level dashboards.',
        phase3: 'Clinical validation study across 500 patients at 3 hospitals, FDA 510(k) submission as a Class II decision support tool, peer-reviewed publication, and hospital enterprise contracts.',
    },
    demo: {
        tagline: 'See it in action',
        points: [
            '"25% of physicians are over 60. They\'re missing crackles at 2–5 kHz. We\'re giving stethoscopes superhuman hearing."',
            'Open RespiLens in your browser. No installation. No app store. Just a URL that works instantly.',
            'Record or upload a lung sound — within seconds, see the classification: sound type, confidence, severity, and the acoustic evidence behind it.',
            'Explore patient history — 5 recordings over 2 weeks, severity trending from 32 to 68. This patient is deteriorating. A single visit might not reveal that.',
            'Today: instant web access. Next: EHR integration and native apps. Long-term: FDA-cleared clinical decision support.',
        ],
    },
    team: [
        { name: 'Hari Manivannan', role: 'Project Lead / Full-Stack' },
        { name: 'Member 2', role: 'Machine Learning / AI' },
        { name: 'Member 3', role: 'Frontend / UX Design' },
    ],
}
