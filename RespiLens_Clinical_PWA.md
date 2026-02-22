# RespiLens Clinical PWA

> *"Accessible via URL. Clinical-grade quality. No compromise."*

---

## 1. Executive Summary

25% of primary care physicians in the United States are over 60 years old. Presbycusis — age-related hearing loss — begins at 40 and disproportionately affects the exact frequency range (2–5kHz) where fine crackles occur. A 65-year-old family doctor examining a pneumonia patient may genuinely miss early bilateral crackles that a younger clinician would detect. This is not incompetence. It is physiology.

RespiLens Clinical solves this by applying deep learning to lung sound recordings captured with digital stethoscopes. The system classifies adventitious sounds (crackles vs wheezes, fine vs coarse, monophonic vs polyphonic), identifies their timing in the respiratory cycle (inspiratory vs expiratory), and provides SHAP-based explainability so clinicians understand exactly which acoustic features drove the classification.

We built RespiLens as a Progressive Web App because clinical tools need to be accessible, not locked behind app stores. A physician in a rural clinic can use this tomorrow by opening a URL — no IT approval, no MDM enrollment, no installation. But we are not compromising on quality — this uses the Web Audio API for clinical-grade signal processing, runs entirely in the browser, and works offline via service workers.

---

## 2. Problem Statement

### 2.1 Physician Hearing Limitations

Auscultation is the cornerstone of respiratory diagnosis. But the human ear is a biological instrument with measurable limitations that degrade with age.

| Physician Age | Typical Hearing Loss | Clinical Impact |
|---|---|---|
| 40–50 years | Subtle loss >4kHz | May miss fine crackles |
| 50–60 years | Moderate loss >3kHz | Crackles harder to detect |
| 60+ years | Significant loss >2kHz | May rely on wheeze only |

Fine crackles — the earliest indicator of pulmonary fibrosis, interstitial lung disease, and early-stage pneumonia — occur at 2–5kHz. This is precisely the range that presbycusis eliminates first.

### 2.2 Clinical Accessibility Gap

Medical software distribution is broken. Physicians need IT approval to install software. Hospital MDM policies block app downloads. App store reviews take weeks. Meanwhile, patients wait.

A web application eliminates every barrier. No download. No installation. No app store. Just a URL that works instantly on any device with a modern browser.

---

## 3. Solution Overview

### 3.1 Why a Progressive Web App

We built RespiLens Clinical as a Progressive Web App (PWA) — not a native app — for three strategic reasons:

- **Instant deployment** — physicians access it via URL, no installation required
- **Universal compatibility** — works on desktop, tablet, and mobile without separate codebases
- **Clinical accessibility** — bypasses hospital IT restrictions and app store delays

A PWA is not a compromise. It is a strategic advantage. Physicians can use RespiLens in clinic tomorrow by opening respi-lens.app — no IT ticket, no MDM enrollment, no waiting. For clinical tools, accessibility is more important than native polish.

### 3.2 How It Works — Physician Workflow

| # | Step | Detail |
|---|---|---|
| 1 | Open URL | Physician navigates to respi-lens.app in Chrome or Edge. Instant access. No login required for demo mode. |
| 2 | Connect Stethoscope | Bluetooth digital stethoscope pairs via Web Bluetooth API. Alternative: wired input via 3.5mm microphone adapter or upload pre-recorded .wav file. |
| 3 | Record Auscultation | Physician auscultates patient for 10–30 seconds. Web Audio API captures audio at 44.1kHz. Live waveform displayed via Canvas API. |
| 4 | Upload to Backend | Audio uploaded to Supabase Storage. FastAPI backend retrieves via signed URL and begins processing. |
| 5 | Real-Time Analysis | Respiratory cycles extracted. Features computed. Multi-task CNN runs inference. SHAP values computed. |
| 6 | Classification Result | Result pushed to frontend via Supabase Realtime. Displays: sound type (crackle/wheeze/both/normal), disease probability, severity score, SHAP attribution. |
| 7 | Clinical Decision | Physician reviews AI classification alongside their own findings. Can download PDF report or save to patient record. |

---

## 4. Training Data & Datasets

### 4.1 Primary Dataset: ICBHI 2017

ICBHI 2017 Respiratory Sound Database — 6,898 annotated respiratory cycles from 126 patients. Freely available at https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

| Attribute | Details |
|---|---|
| Total Duration | 5.5 hours of recordings |
| Respiratory Cycles | 6,898 cycles annotated by respiratory medicine experts |
| Patients | 126 subjects (age 0–91, mean 49.6 years) |
| Sound Labels | Normal (3,642 cycles), Crackles only (1,864), Wheezes only (886), Both (506) |
| Disease Labels | COPD, Pneumonia, Bronchitis, Asthma, Bronchiectasis, URTI, LRTI, Healthy |
| Recording Devices | Multiple stethoscope types to ensure device-agnostic training |
| Chest Locations | 7 positions: Anterior left/right, Posterior left/right, Lateral left/right, Trachea |

### 4.2 Supplementary Datasets

- **PKU Respiratory Dataset** — 11,968 cycles, includes COVID-19 pneumonia cases
- **HF_Lung_V2** — Extended labels for respiratory cycle phase (inspiratory vs expiratory)
- **Mendeley Lung Sounds** — 112 patients, multiple chest positions and filter settings

### 4.3 Training Strategy

**Hackathon MVP:** ICBHI 2017 only. Target: 70–75% sound classification accuracy.

**Post-Hackathon:** Combine ICBHI + PKU for +8% accuracy gain and better generalization.

---

## 5. Machine Learning Pipeline

### 5.1 Feature Extraction

Lung sounds occupy 50–2000Hz. Features isolate crackles (2–5kHz) and wheezes (100–1000Hz) from normal breathing.

| Category | Features | Clinical Relevance |
|---|---|---|
| Spectral | MFCCs (20), spectral centroid, bandwidth, rolloff | Crackles create sharp peaks at 2–5kHz |
| Time-Frequency | Mel-spectrogram (128 bins, 50–2000Hz) | Primary CNN input — reveals temporal evolution |
| Zero-Crossing Rate | ZCR mean, std, variance | Crackles have extremely high ZCR |
| Energy-Based | RMS energy, short-time energy | Crackles are high-energy transients |
| Harmonic Content | HNR, spectral flux | Wheezes have high harmonic content |

### 5.2 Model Architecture

Multi-task CNN with dual input branches (spectrogram + tabular features) and three output heads (sound classification, disease probability, severity score).

### 5.3 SHAP Explainability

Every classification includes SHAP values showing which features drove the decision. Top 5 features displayed with plain-language labels.

---

## 6. Technical Stack

### 6.1 Frontend — Progressive Web App

| Technology | Version | Role |
|---|---|---|
| React | 18.2 | UI framework — component-based architecture |
| Vite | 5.0 | Build tool and dev server — instant hot reload |
| TypeScript | 5.3 | Type safety across entire codebase |
| Tailwind CSS | 3.4 | Utility-first styling |
| shadcn/ui | latest | Accessible component primitives |
| Recharts | 2.10 | 7-day trend chart visualization |
| Framer Motion | 10.16 | Score reveal animations |
| Zustand | 4.5 | Lightweight global state management |
| vite-plugin-pwa | 0.17 | Auto-generate PWA manifest + service workers |
| workbox | 7.0 | Service worker caching strategies |

### 6.2 Audio Capture — Web Audio API

Web Audio API provides clinical-grade audio processing entirely in the browser — no native code required. Supports real-time FFT analysis, waveform visualization, and high-quality recording at 44.1kHz sample rate.

| API/Feature | Usage in RespiLens |
|---|---|
| `navigator.mediaDevices` | Access microphone or Bluetooth audio input. Requests user permission once, then records. |
| `AudioContext` | Core Web Audio API object. Sample rate: 44,100Hz (CD quality, sufficient for lung sounds <2kHz). |
| `AnalyserNode` | Real-time FFT for waveform visualization. 2048-point FFT provides frequency resolution for crackle detection. |
| `MediaRecorder API` | Records audio to .webm or .wav format for backend upload. Auto-chunks every 1 second for progress indication. |
| `Canvas API` | Renders live waveform during recording. 60fps animation shows physician that audio is being captured. |
| `Web Bluetooth API` | Pairs with Bluetooth stethoscopes (Eko, Stemoscope). Chrome/Edge only — Safari not supported. |

### 6.3 PWA Configuration

```typescript
// vite.config.ts
import { VitePWA } from 'vite-plugin-pwa'

export default {
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      manifest: {
        name: 'RespiLens Clinical',
        short_name: 'RespiLens',
        description: 'AI-powered lung sound analysis',
        theme_color: '#4F46E5',
        background_color: '#0F172A',
        display: 'standalone',  // Opens fullscreen like native app
        icons: [
          { src: '/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icon-512.png', sizes: '512x512', type: 'image/png' }
        ]
      },
      workbox: {
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\..*/,
            handler: 'NetworkFirst',  // Try network, fallback to cache
            options: { cacheName: 'api-cache' }
          }
        ]
      }
    })
  ]
}
```

### 6.4 Backend — FastAPI ML Server

| Technology | Version | Role |
|---|---|---|
| Python | 3.11 | Primary backend language |
| FastAPI | 0.104 | Async REST API for ML inference |
| Uvicorn | 0.24 | ASGI server |
| supabase-py | 2.0 | Python client for database and storage |
| librosa | 0.10 | Audio feature extraction |
| scipy | 1.11 | Signal processing |
| PyTorch | 2.1 | Multi-task CNN training and inference |
| shap | 0.43 | SHAP explainability |
| onnxruntime | 1.16 | Optimized inference (no PyTorch at runtime) |

### 6.5 Supabase — Data Layer

| Service | Usage |
|---|---|
| Auth (Anonymous) | Zero-friction physician onboarding — no email required for demo |
| PostgreSQL | Stores patient recordings, classifications, SHAP data |
| Storage | Audio files in private buckets. Auto-deleted after 90 days. |
| Realtime | Classification results pushed to frontend without polling |
| PostgREST API | Auto-generated REST API for patient history queries |

### 6.6 Infrastructure & Deployment

| Service | Role |
|---|---|
| Vercel | Frontend deployment. Auto-HTTPS, global CDN, instant deployment from git push. |
| Railway | FastAPI backend deployment. Persistent disk for ONNX model. |
| Supabase Cloud | Managed PostgreSQL + Auth + Storage + Realtime. Free tier handles hackathon load. |
| Docker | Containerize FastAPI for local development and Railway deployment. |
| GitHub Actions | Run pytest on every push to validate inference pipeline. |

---

## 7. Demo Plan

### 7.1 The 5-Minute Demo Script

| Time | Act | What You Show |
|---|---|---|
| 0:00 – 0:45 | The Problem | "25% of primary care physicians are over 60. They are missing crackles at 2–5kHz. We are giving stethoscopes superhuman hearing." |
| 0:45 – 1:15 | Demo Begins | Open laptop. Navigate to respi-lens.app. Instant load. "No installation. No app store. Just a URL." Hand digital stethoscope to judge. |
| 1:15 – 2:00 | Live Recording | Judge places stethoscope on volunteer chest. Click Record. Live waveform animates. 10 seconds. "Did you hear crackles?" Judge says no. |
| 2:00 – 2:45 | Classification Reveal | Result appears: "Fine crackles detected. Bilateral. Inspiratory. 87% confidence." SHAP card: "High spectral energy at 2.3kHz." Point to waveform showing exact crackle timing. |
| 2:45 – 3:30 | The Depth | Switch to patient history. Show pre-seeded trend: 5 recordings over 2 weeks, severity rising 32→68. "This patient is deteriorating. Physician might not notice without trend data." |
| 3:30 – 4:00 | Clinical Report | Click Download Report. PDF generates in 2 seconds. "This goes into EHR or gets faxed to specialist." |
| 4:00 – 5:00 | The Vision | "Today: instant web access. 6 months: native app + EHR integration. 18 months: FDA 510(k). We built for accessibility first. We are not compromising on quality." |

### 7.2 Demo URL Strategy

Deploy to Vercel with custom domain: respi-lens.app. Short, memorable, professional. Judges type it on their own devices during deliberation.

### 7.3 Fallback Scenarios

- **Bluetooth fails:** Use wired 3.5mm microphone input
- **Live processing fails:** Pre-computed results served from `/api/demo/{scenario}`
- **Internet fails:** Service worker caches entire app, works offline

---

## 8. Product Roadmap

### Phase 1 — Hackathon MVP (0–48 hours)

| Feature | Status | Notes |
|---|---|---|
| Web Audio API capture (mic or Bluetooth) | Build | Chrome/Edge only, Safari fallback to file upload |
| Live waveform visualization (Canvas API) | Build | 60fps animation during recording |
| Lung sound classification (4 classes) | Build | Trained on ICBHI 2017 |
| SHAP explainability (top 5 features) | Build | Plain-language labels |
| Real-time result via Supabase Realtime | Build | No polling, instant push |
| 7-day patient trend chart (Recharts) | Build | Longitudinal severity visualization |
| PDF clinical report download | Build | Browser-native PDF generation |
| PWA manifest + service workers | Build | Offline capability, Add to Home Screen |

### Phase 2 — Clinical Beta (Months 1–3)

- Native mobile apps (React Native) — iOS and Android for physicians who prefer apps
- Multi-dataset training — ICBHI + PKU for +8% accuracy
- EHR integration pilot — FHIR export to Epic, Cerner
- Teaching mode for medical students — annotated playback with quiz
- Clinic dashboard — aggregate analytics for clinic admins

### Phase 3 — FDA Pathway (Months 4–12)

- Clinical validation study — 500 patients across 3 hospitals
- FDA 510(k) application — Class II decision support tool
- Published peer-reviewed paper — validation results
- Hospital enterprise contracts — deploy at 5 health systems

---

## 9. Business Model

### 9.1 Pricing Tiers

| Tier | Price | Target | Included |
|---|---|---|---|
| Individual | $79/month | Solo practitioners | Unlimited recordings, SHAP, PDF reports |
| Clinic (5 seats) | $199/month | Small practices | All individual + clinic dashboard |
| Hospital | $999/month | Health systems | Unlimited physicians + EHR integration |
| Student | $19/month | Med students | All features + teaching mode |

### 9.2 Market Size

**TAM:** 1.1M active physicians in US.
**SAM:** 320K (physicians 50+, med students, telemedicine-focused).
**SOM (Year 3):** 1% = 3,200 users × $50 ARPU = $1.92M ARR.

---

## 10. Team & Timeline

### 10.1 Hour-by-Hour Schedule

| Hours | Milestone |
|---|---|
| 0 – 2 | **SETUP.** Repo created, Vite React project scaffolded, Supabase + Railway configured, ICBHI dataset downloaded, digital stethoscope tested. |
| 2 – 6 | **FOUNDATION.** ML: ICBHI data loaded, features extracted. Backend: Supabase connected, /api/analyze stub. Frontend: Web Audio API recording working. |
| 6 – 12 | **CORE.** ML: model training started. Backend: feature extraction endpoint live. Frontend: recording → upload → POST → mock result displayed. |
| 12 – 18 | **INTEGRATION.** ML: model trained, ONNX exported. Backend: real inference working. Frontend: Realtime subscription working, SHAP cards rendered. |
| 18 – 22 | **POLISH.** Frontend: waveform annotations, trend chart, PDF download. Backend: disease probability added. PWA manifest configured. |
| 22 – 24 | **DEMO PREP.** Deploy to Vercel (respi-lens.app). Test on 3 browsers. Rehearse demo script. Pre-recorded samples loaded as backup. |
