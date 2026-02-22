import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { DashboardPage } from './pages/DashboardPage'
import { RecordingPage } from './pages/RecordingPage'
import { PatientHistoryPage } from './pages/PatientHistoryPage'
import { AnalysisDetailPage } from './pages/AnalysisDetailPage'
import { AboutPage } from './pages/AboutPage'

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<DashboardPage />} />
        <Route path="record" element={<RecordingPage />} />
        <Route path="history" element={<PatientHistoryPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="analysis/:id" element={<AnalysisDetailPage />} />
      </Route>
    </Routes>
  )
}

export default App
