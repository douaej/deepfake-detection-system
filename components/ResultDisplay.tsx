'use client'

import { useState, useEffect } from 'react'
import { AnalysisResult, isImageResult, isVideoResult, DetectionResult } from '@/types'

interface ResultDisplayProps {
  result: AnalysisResult
  fileName?: string
}

interface HistoryEntry {
  id: string; fileName: string; verdict: DetectionResult
  confidence: number; timestamp: string
}

let _history: HistoryEntry[] = []

// ── PDF Export ─────────────────────────────────────────────────────────────
function exportPDF(result: AnalysisResult, fileName: string) {
  const isImg = isImageResult(result)
  const verdict = isImg ? result.result : (result as any).overall_result
  const confidence = isImg ? result.confidence : (result as any).avg_confidence
  const isFake = verdict === 'FAKE'
  const explanation = isImg ? (result as any).explanation : null
  const heatmap = isImg ? result.evidence?.heatmap_base64 : null
  const now = new Date().toLocaleString('fr-FR')

  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8"/><title>Rapport Deepfake</title>
<style>*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Courier New',monospace;background:#06090d;color:#ddeeff;padding:40px;font-size:13px;}
.hd{border-bottom:1px solid #1a2a3a;padding-bottom:18px;margin-bottom:26px;}
.ht{font-size:18px;font-weight:800;letter-spacing:.04em;font-family:Georgia,serif;}
.hs{color:#6a90aa;font-size:10px;margin-top:4px;letter-spacing:.1em;}
.vb{border:1px solid;padding:18px 22px;margin-bottom:18px;position:relative;}
.vf{background:rgba(255,51,85,.06);border-color:rgba(255,51,85,.3);}
.vr{background:rgba(0,255,136,.06);border-color:rgba(0,255,136,.3);}
.vl{font-size:22px;font-weight:900;letter-spacing:.03em;}
.vf .vl{color:#ff3355;}.vr .vl{color:#00ff88;}
.vc{font-size:11px;color:#6a90aa;margin-top:6px;}
.bg{background:#1a2a3a;height:3px;margin-top:6px;}
.bf{height:100%;}.bfk{background:#ff3355;}.brl{background:#00ff88;}
.sec{margin-bottom:18px;}
.st{font-size:8px;letter-spacing:.18em;color:#2d4a5e;text-transform:uppercase;margin-bottom:7px;border-bottom:1px solid #1a2a3a;padding-bottom:5px;}
.ex{background:#0b1018;border:1px solid #1a2a3a;border-left:2px solid #00aacc;padding:11px 13px;font-size:11px;line-height:1.8;color:#6a90aa;}
.sg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}
.sb{background:#0b1018;border:1px solid #1a2a3a;padding:10px;}
.sl{font-size:8px;letter-spacing:.13em;text-transform:uppercase;color:#2d4a5e;}
.sv{font-size:17px;font-weight:800;margin-top:3px;}
.hm{width:100%;border:1px solid #1a2a3a;margin-top:6px;}
.ft{margin-top:32px;border-top:1px solid #1a2a3a;padding-top:13px;color:#2d4a5e;font-size:9px;display:flex;justify-content:space-between;letter-spacing:.08em;}
</style></head><body>
<div class="hd"><div class="ht">DEEPFAKE_DETECTOR / RAPPORT D'ANALYSE</div><div class="hs">GÉNÉRÉ ${now} · ${fileName}</div></div>
<div class="vb v${isFake?'f':'r'}">
<div class="vl">${isFake?'[ DEEPFAKE DÉTECTÉ ]':'[ IMAGE AUTHENTIQUE ]'}</div>
<div class="vc">CONFIANCE : ${Math.round(confidence*100)}%</div>
<div class="bg"><div class="bf b${isFake?'fk':'rl'}" style="width:${Math.round(confidence*100)}%"></div></div>
</div>
${explanation?`<div class="sec"><div class="st">ANALYSE EXPLICATIVE</div><div class="ex">${explanation}</div></div>`:''}
${isImg?`<div class="sec"><div class="st">MÉTRIQUES</div><div class="sg">
<div class="sb"><div class="sl">PROB. FAKE</div><div class="sv">${Math.round((result as any).fake_probability*100)}%</div></div>
<div class="sb"><div class="sl">VISAGES</div><div class="sv">${result.evidence?.num_faces??0}</div></div>
<div class="sb"><div class="sl">TEMPS</div><div class="sv">${Math.round(result.processing_time_ms)}ms</div></div>
</div></div>`:''}
${heatmap?`<div class="sec"><div class="st">ACTIVATION MAP / GRADCAM</div><img class="hm" src="data:image/png;base64,${heatmap}"/></div>`:''}
<div class="ft"><span>DEEPFAKE DETECTION · EFFICIENTNET-B0 · GRADCAM</span><span>SEUIL : 50%</span></div>
</body></html>`

  const blob = new Blob([html],{type:'text/html'})
  const url = URL.createObjectURL(blob)
  const win = window.open(url,'_blank')
  if(win) win.onload=()=>setTimeout(()=>{win.print();URL.revokeObjectURL(url)},500)
}

// ── Confidence bar ─────────────────────────────────────────────────────────
function ConfBar({value, isFake}:{value:number; isFake:boolean}) {
  const pct = Math.round(Math.max(0,Math.min(1,isNaN(value)?0:value))*100)
  const color = isFake ? 'var(--red)' : 'var(--green)'
  return (
    <div>
      <div style={{display:'flex',justifyContent:'space-between',marginBottom:6}}>
        <span style={{color:'var(--text-faint)',fontSize:9,letterSpacing:'0.16em'}}>CONFIANCE</span>
        <span style={{color,fontSize:12,fontWeight:700,fontFamily:'var(--mono)'}}>{pct}%</span>
      </div>
      <div style={{height:3,background:'var(--border)',overflow:'hidden',position:'relative'}}>
        <div style={{
          position:'absolute',inset:'0 auto 0 0',
          width:`${pct}%`,
          background:`linear-gradient(90deg, ${color}66, ${color})`,
          boxShadow:`0 0 10px ${color}`,
          transition:'width 0.9s cubic-bezier(0.22,1,0.36,1)',
        }}/>
      </div>
    </div>
  )
}

// ── Stat cell ──────────────────────────────────────────────────────────────
function Cell({label,value,accent}:{label:string;value:string;accent?:boolean}) {
  return (
    <div style={{
      border:`1px solid ${accent?'rgba(0,229,255,0.25)':'var(--border)'}`,
      background:accent?'rgba(0,229,255,0.04)':'var(--surface)',
      padding:'11px 13px',
    }}>
      <div style={{color:'var(--text-faint)',fontSize:8,letterSpacing:'0.16em',marginBottom:5}}>{label}</div>
      <div style={{color:accent?'var(--cyan)':'#fff',fontSize:15,fontWeight:700,fontFamily:'var(--mono)'}}>{value}</div>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────
export default function ResultDisplay({ result, fileName = 'fichier' }: ResultDisplayProps) {
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [showHistory, setShowHistory] = useState(false)

  const isImg = isImageResult(result)
  const verdict: DetectionResult = isImg ? result.result : (result as any).overall_result
  const confidence = isImg ? result.confidence : (result as any).avg_confidence
  const isFake  = verdict === 'FAKE'
  const isReal  = verdict === 'REAL'
  const explanation = isImg ? (result as any).explanation : null
  const heatmap     = isImg ? result.evidence?.heatmap_base64 : null
  const noFace      = isImg && result.evidence?.num_faces === 0

  const accent     = isFake ? 'var(--red)'   : isReal ? 'var(--green)'   : 'var(--amber)'
  const accentGlow = isFake ? 'var(--red-glow)' : isReal ? 'var(--green-glow)' : 'rgba(255,170,0,0.08)'

  useEffect(() => {
    const entry: HistoryEntry = {
      id: Date.now().toString(), fileName, verdict, confidence,
      timestamp: new Date().toLocaleTimeString('fr-FR'),
    }
    _history = [entry, ..._history].slice(0, 20)
    setHistory([..._history])
  }, [result])

  if (!isImageResult(result) && !isVideoResult(result)) return null

  return (
    <div className="fade-up" style={{ display:'flex', flexDirection:'column', gap:12 }}>

      {/* Header row */}
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <div style={{ width:2, height:13, background:'var(--cyan)' }}/>
          <span style={{ color:'var(--text-dim)', fontSize:9, letterSpacing:'0.16em' }}>
            OUTPUT / RÉSULTAT D'ANALYSE
          </span>
        </div>
        <button
          onClick={() => exportPDF(result, fileName)}
          style={{
            fontFamily:'var(--mono)', fontSize:9, letterSpacing:'0.1em',
            color:'var(--text-dim)', border:'1px solid var(--border)',
            background:'transparent', padding:'4px 10px', cursor:'pointer',
            transition:'color 0.2s, border-color 0.2s',
          }}
          onMouseEnter={e=>{(e.currentTarget.style.color='var(--cyan)');(e.currentTarget.style.borderColor='var(--cyan)')}}
          onMouseLeave={e=>{(e.currentTarget.style.color='var(--text-dim)');(e.currentTarget.style.borderColor='var(--border)')}}
        >
          ↓ RAPPORT PDF
        </button>
      </div>

      {/* Verdict card */}
      <div style={{
        border:`1px solid ${accent}`,
        background: accentGlow,
        padding:'22px', position:'relative', overflow:'hidden',
        boxShadow:`0 0 40px ${accentGlow}, inset 0 1px 0 ${accent}22`,
      }}>
        {/* Decorative triangle corner */}
        <div style={{
          position:'absolute', top:0, right:0,
          width:0, height:0,
          borderTop:`50px solid ${accent}15`,
          borderLeft:'50px solid transparent',
        }}/>

        {/* File label */}
        <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:12 }}>
          <div style={{ width:5, height:5, background:accent, boxShadow:`0 0 8px ${accent}` }}/>
          <span style={{ color:'var(--text-faint)', fontSize:9, letterSpacing:'0.12em' }}>
            {isImg?'IMAGE':'VIDEO'} · {fileName}
          </span>
        </div>

        {/* Verdict text */}
        <div style={{
          fontFamily:'var(--display)', fontSize:clamp(26), fontWeight:800,
          color:accent, letterSpacing:'-0.01em', lineHeight:1,
          textShadow:`0 0 40px ${accent}`,
          marginBottom:16,
        }}>
          {isFake ? '[ DEEPFAKE ]' : isReal ? '[ AUTHENTIQUE ]' : '[ INCONNU ]'}
        </div>

        <ConfBar value={confidence} isFake={isFake}/>

        {/* No face warning */}
        {noFace && (
          <div style={{
            marginTop:12, padding:'7px 11px',
            border:'1px solid rgba(255,170,0,0.3)',
            background:'rgba(255,170,0,0.06)',
            fontSize:9, color:'var(--amber)', letterSpacing:'0.05em',
          }}>
            ⚠ Aucun visage détecté — analyse réalisée sur l'image entière
          </div>
        )}
      </div>

      {/* Stats grid */}
      {isImg && (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:8 }}>
          <Cell label="VERDICT" value={verdict} accent/>
          <Cell label="PROB. FAKE" value={`${Math.round(result.fake_probability*100)}%`}/>
          <Cell label="VISAGES" value={String(result.evidence?.num_faces??0)}/>
        </div>
      )}
      {isVideoResult(result) && (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:8 }}>
          <Cell label="VERDICT" value={verdict} accent/>
          <Cell label="RATIO FAKE" value={`${Math.round(((result as any).stats?.fake_ratio??0)*100)}%`}/>
          <Cell label="FRAMES" value={String((result as any).stats?.num_frames??0)}/>
        </div>
      )}

      {/* Explanation */}
      {explanation && (
        <div style={{
          border:'1px solid var(--border)', borderLeft:'2px solid var(--cyan-dim)',
          background:'var(--surface)', padding:'12px 14px',
        }}>
          <div style={{ color:'var(--text-faint)', fontSize:8, letterSpacing:'0.16em', marginBottom:6 }}>
            ANALYSE EXPLICATIVE
          </div>
          <p style={{ color:'var(--text-dim)', fontSize:11, lineHeight:1.95 }}>{explanation}</p>
        </div>
      )}

      {/* GradCAM heatmap */}
      {heatmap && (
        <div style={{ border:'1px solid var(--border)', background:'var(--surface)', overflow:'hidden' }}>
          <div style={{
            padding:'8px 12px', borderBottom:'1px solid var(--border)',
            display:'flex', alignItems:'center', gap:8,
          }}>
            <div style={{ width:2, height:10, background:'var(--cyan)' }}/>
            <span style={{ color:'var(--text-faint)', fontSize:8, letterSpacing:'0.14em' }}>
              ACTIVATION MAP / GRADCAM
            </span>
          </div>
          <img
            src={`data:image/png;base64,${heatmap}`}
            alt="GradCAM Heatmap"
            style={{ width:'100%', display:'block' }}
          />
          <div style={{ padding:'6px 12px' }}>
            <span style={{ color:'var(--text-faint)', fontSize:8, letterSpacing:'0.04em' }}>
              // zones chaudes = régions déterminantes pour la classification
            </span>
          </div>
        </div>
      )}

      {/* Timing */}
      <div style={{
        display:'flex', justifyContent:'space-between', alignItems:'center',
        padding:'7px 12px', border:'1px solid var(--border)', background:'var(--surface)',
      }}>
        <span style={{ color:'var(--text-faint)', fontSize:8, letterSpacing:'0.12em' }}>TEMPS DE TRAITEMENT</span>
        <span style={{ color:'var(--cyan-dim)', fontSize:11, fontFamily:'var(--mono)' }}>
          {isImg
            ? `${Math.round(result.processing_time_ms)}ms`
            : `${((result as any).processing_time_ms/1000).toFixed(1)}s`}
        </span>
      </div>

      {/* History */}
      {history.length > 1 && (
        <div style={{ border:'1px solid var(--border)', background:'var(--surface)' }}>
          <button
            onClick={() => setShowHistory(v=>!v)}
            style={{
              width:'100%', fontFamily:'var(--mono)',
              display:'flex', alignItems:'center', justifyContent:'space-between',
              padding:'9px 12px', background:'transparent', border:'none',
              cursor:'pointer', color:'var(--text-dim)',
            }}
          >
            <div style={{ display:'flex', alignItems:'center', gap:8 }}>
              <div style={{ width:2, height:10, background:'var(--border2)' }}/>
              <span style={{ fontSize:9, letterSpacing:'0.12em' }}>
                HISTORIQUE SESSION ({history.length} ANALYSES)
              </span>
            </div>
            <span style={{ fontSize:9 }}>{showHistory?'▲':'▼'}</span>
          </button>

          {showHistory && (
            <div style={{ borderTop:'1px solid var(--border)' }}>
              {history.map(e => (
                <div key={e.id} style={{
                  display:'flex', alignItems:'center', justifyContent:'space-between',
                  padding:'7px 12px', borderBottom:'1px solid var(--border)',
                }}>
                  <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                    <div style={{
                      width:5, height:5, flexShrink:0,
                      background: e.verdict==='FAKE'?'var(--red)':'var(--green)',
                      boxShadow:`0 0 5px ${e.verdict==='FAKE'?'var(--red)':'var(--green)'}`,
                    }}/>
                    <span style={{ color:'var(--text-dim)', fontSize:10, maxWidth:155, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                      {e.fileName}
                    </span>
                  </div>
                  <div style={{ display:'flex', gap:10, alignItems:'center', flexShrink:0 }}>
                    <span style={{ fontSize:9, fontWeight:700, color:e.verdict==='FAKE'?'var(--red)':'var(--green)' }}>
                      {e.verdict}
                    </span>
                    <span style={{ color:'var(--text-faint)', fontSize:9 }}>{Math.round(e.confidence*100)}%</span>
                    <span style={{ color:'var(--text-faint)', fontSize:9 }}>{e.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// helper — évite d'importer clamp
function clamp(n: number) { return n }