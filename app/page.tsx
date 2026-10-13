'use client'

import { useState } from 'react'
import Header from '@/components/Header'
import FileUpload from '@/components/FileUpload'
import ResultDisplay from '@/components/ResultDisplay'
import { AnalysisResult, DetectionResult } from '@/types'

function transformResponse(data: any, isImage: boolean): AnalysisResult {
  console.log('[v0] Raw API response:', data)
  if (isImage) {
    return {
      result: data.result || 'UNKNOWN',
      confidence: typeof data.confidence === 'number' ? data.confidence : 0,
      fake_probability: typeof data.fake_probability === 'number' ? data.fake_probability : 0,
      processing_time_ms: data.processing_time_ms || 0,
      evidence: data.evidence || {},
      request_id: data.request_id,
    }
  } else {
    const overallResult: DetectionResult = data.overall_result || 'UNKNOWN'
    const avgConfidence = typeof data.avg_confidence === 'number'
      ? data.avg_confidence
      : (data.stats?.mean_confidence ?? 0)
    return {
      overall_result: overallResult,
      avg_confidence: avgConfidence,
      stats: {
        num_frames: data.stats?.num_frames ?? 0,
        fake_ratio: data.stats?.fake_ratio ?? 0,
        mean_confidence: data.stats?.mean_confidence,
        std_confidence: data.stats?.std_confidence,
        min_confidence: data.stats?.min_confidence,
        max_confidence: data.stats?.max_confidence,
        fake_count: data.stats?.fake_count,
      },
      processing_time_ms: data.processing_time_ms || 0,
      request_id: data.request_id,
    }
  }
}

export default function Page() {
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('fichier')

  const handleFileAnalysis = async (file: File) => {
    setLoading(true)
    setError(null)
    setFileName(file.name)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const apiBase = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:5000'
      const isImage = file.type.startsWith('image/')
      const endpoint = isImage ? '/api/detect/image' : '/api/detect/video'
      const response = await fetch(`${apiBase}${endpoint}`, { method: 'POST', body: formData })
      if (!response.ok) {
        let message = `API Error ${response.status}: ${response.statusText}`
        try { const err = await response.json(); if (err?.error) message = err.error } catch {}
        throw new Error(message)
      }
      const data = await response.json()
      setResult(transformResponse(data, isImage))
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur lors de l'analyse")
      console.error('Analysis error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}>
      <Header />

      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '36px 28px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28, alignItems: 'start' }}>

          {/* Left — upload */}
          <FileUpload onAnalyze={handleFileAnalysis} loading={loading} disabled={loading} />

          {/* Right — result */}
          <div>
            {error && (
              <div style={{
                border: '1px solid rgba(255,51,85,0.4)', borderLeft: '2px solid var(--red)',
                background: 'rgba(255,51,85,0.06)', padding: '12px 16px', marginBottom: 14,
              }}>
                <div style={{ color: 'var(--text-faint)', fontSize: 8, letterSpacing: '0.16em', marginBottom: 5 }}>
                  ERREUR SYSTÈME
                </div>
                <p style={{ color: '#ff8899', fontSize: 11, lineHeight: 1.7 }}>{error}</p>
              </div>
            )}

            {result ? (
              <ResultDisplay result={result} fileName={fileName} />
            ) : !error ? (
              <div style={{
                border: '1px solid var(--border)', background: 'var(--surface)',
                padding: '56px 24px', textAlign: 'center', position: 'relative', overflow: 'hidden',
              }}>
                {/* Crosshair */}
                {[120, 60].map(size => (
                  <div key={size} style={{
                    position: 'absolute', top: '50%', left: '50%',
                    transform: 'translate(-50%,-50%)',
                    width: size, height: size,
                    border: '1px solid var(--border)', borderRadius: '50%', opacity: 0.4,
                  }}/>
                ))}
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: 1, height: 80, background: 'var(--border)' }}/>
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: 80, height: 1, background: 'var(--border)' }}/>

                <div style={{ position: 'relative', zIndex: 1 }}>
                  <p style={{ fontFamily: 'var(--display)', fontSize: 13, fontWeight: 700, color: 'var(--text-faint)', letterSpacing: '0.06em' }}>
                    EN ATTENTE D'UN FICHIER
                  </p>
                  <p style={{ color: 'var(--text-faint)', fontSize: 9, marginTop: 8, letterSpacing: '0.08em' }}>
                    // déposer une image ou vidéo pour lancer l'analyse
                  </p>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </main>

      <style>{`
        @media (max-width: 768px) {
          main > div { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  )
}