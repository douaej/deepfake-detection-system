'use client'

import { useState, useRef } from 'react'

interface FileUploadProps {
  onAnalyze: (file: File) => void
  loading: boolean
  disabled: boolean
}

const ALLOWED_IMAGE_TYPES = ['image/jpeg','image/png','image/gif','image/bmp','image/webp']
const ALLOWED_VIDEO_TYPES = ['video/mp4','video/avi','video/quicktime','video/x-mkv']
const ALL_TYPES = [...ALLOWED_IMAGE_TYPES, ...ALLOWED_VIDEO_TYPES]

// Corner bracket decoration
function Corners({ active }: { active: boolean }) {
  const c = active ? 'var(--cyan)' : 'var(--border2)'
  const s = { position:'absolute' as const, width:12, height:12, transition:'border-color 0.2s' }
  return <>
    <div style={{ ...s, top:0, left:0,  borderTop:`2px solid ${c}`, borderLeft:`2px solid ${c}` }}/>
    <div style={{ ...s, top:0, right:0, borderTop:`2px solid ${c}`, borderRight:`2px solid ${c}` }}/>
    <div style={{ ...s, bottom:0, left:0,  borderBottom:`2px solid ${c}`, borderLeft:`2px solid ${c}` }}/>
    <div style={{ ...s, bottom:0, right:0, borderBottom:`2px solid ${c}`, borderRight:`2px solid ${c}` }}/>
  </>
}

export default function FileUpload({ onAnalyze, loading, disabled }: FileUploadProps) {
  const [drag, setDrag] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation()
    setDrag(e.type === 'dragenter' || e.type === 'dragover')
  }
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation()
    setDrag(false)
    if (e.dataTransfer.files?.[0]) process(e.dataTransfer.files[0])
  }
  const process = (f: File) => {
    if (!ALL_TYPES.includes(f.type)) {
      alert('Format non supporté. JPG, PNG, WebP, MP4, AVI, MOV.')
      return
    }
    setFile(f)
    onAnalyze(f)
  }

  const active = drag || loading

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:14 }}>

      {/* Section label */}
      <div style={{ display:'flex', alignItems:'center', gap:8 }}>
        <div style={{ width:2, height:13, background:'var(--cyan)' }}/>
        <span style={{ color:'var(--text-dim)', fontSize:9, letterSpacing:'0.18em' }}>
          INPUT / SÉLECTION FICHIER
        </span>
      </div>

      {/* Drop zone */}
      <div
        onClick={() => !disabled && inputRef.current?.click()}
        onDragEnter={handleDrag} onDragLeave={handleDrag}
        onDragOver={handleDrag} onDrop={handleDrop}
        style={{
          position: 'relative', overflow: 'hidden',
          border: `1px solid ${active ? 'var(--cyan)' : 'var(--border2)'}`,
          background: active ? 'var(--cyan-glow)' : 'var(--surface)',
          padding: '44px 24px', textAlign: 'center',
          cursor: disabled ? 'not-allowed' : 'pointer',
          transition: 'border-color 0.2s, background 0.2s',
          opacity: disabled && !loading ? 0.55 : 1,
          boxShadow: active ? '0 0 30px rgba(0,229,255,0.08)' : 'none',
        }}
      >
        <Corners active={active} />

        {/* Animated scan line while loading */}
        {loading && (
          <div style={{
            position: 'absolute', left:0, right:0, height:2, top:0,
            background: 'linear-gradient(90deg, transparent 0%, var(--cyan) 50%, transparent 100%)',
            boxShadow: '0 0 12px var(--cyan)',
            animation: 'scanLine 1.8s ease-in-out infinite',
          }}/>
        )}

        <input
          ref={inputRef} type="file" style={{ display:'none' }}
          accept={ALL_TYPES.join(',')}
          onChange={e => e.currentTarget.files?.[0] && process(e.currentTarget.files[0])}
          disabled={disabled}
        />

        <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:14 }}>
          {/* Icon */}
          <div style={{
            width:52, height:52, border:`1px solid ${active?'var(--cyan)':'var(--border2)'}`,
            display:'flex', alignItems:'center', justifyContent:'center',
            background: active ? 'var(--cyan-glow)' : 'transparent',
            transition: 'all 0.3s',
            boxShadow: active ? '0 0 20px rgba(0,229,255,0.2)' : 'none',
          }}>
            {loading
              ? <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                  <circle cx="11" cy="11" r="8" stroke="var(--cyan)" strokeWidth="1.5" strokeDasharray="5 3" className="spin"/>
                  <circle cx="11" cy="11" r="3" fill="var(--cyan)" opacity="0.5"/>
                </svg>
              : <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                  <path d="M11 3v13M7 12l4 4 4-4" stroke="var(--cyan)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M3 17v1a2 2 0 002 2h12a2 2 0 002-2v-1" stroke="var(--text-dim)" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
            }
          </div>

          <div>
            <p style={{
              fontFamily: 'var(--display)', fontSize:15, fontWeight:700,
              color: loading ? 'var(--cyan)' : '#fff',
              letterSpacing: '-0.01em',
            }}>
              {loading ? 'ANALYSE EN COURS...' : 'DÉPOSER UN FICHIER'}
            </p>
            {!loading && (
              <p style={{ color:'var(--text-dim)', fontSize:10, marginTop:5, letterSpacing:'0.06em' }}>
                ou cliquer pour parcourir
              </p>
            )}
          </div>

          {/* Format badges */}
          {!loading && (
            <div style={{ display:'flex', gap:5, flexWrap:'wrap', justifyContent:'center' }}>
              {[
                {label:'JPG', type:'img'},{label:'PNG', type:'img'},{label:'WEBP', type:'img'},
                {label:'MP4', type:'vid'},{label:'AVI', type:'vid'},{label:'MOV', type:'vid'},
              ].map(({label,type}) => (
                <span key={label} style={{
                  padding:'2px 8px', fontSize:8, letterSpacing:'0.12em',
                  border:'1px solid var(--border)',
                  color: type==='img' ? 'var(--cyan-dim)' : 'var(--amber)',
                  background: type==='img' ? 'rgba(0,229,255,0.03)' : 'rgba(255,170,0,0.03)',
                }}>
                  {label}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Selected file info */}
      {file && (
        <div style={{
          border:'1px solid var(--border)', background:'var(--surface)',
          padding:'10px 14px', display:'flex', alignItems:'center', justifyContent:'space-between',
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:10 }}>
            <div style={{
              width:7, height:7, flexShrink:0,
              background: file.type.startsWith('image/') ? 'var(--cyan)' : 'var(--amber)',
              boxShadow: `0 0 8px ${file.type.startsWith('image/') ? 'var(--cyan)' : 'var(--amber)'}`,
            }}/>
            <span style={{ color:'#fff', fontSize:11, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', maxWidth:200 }}>
              {file.name}
            </span>
          </div>
          <div style={{ display:'flex', gap:12, alignItems:'center', flexShrink:0 }}>
            <span style={{ color:'var(--text-dim)', fontSize:9, letterSpacing:'0.1em' }}>
              {file.type.startsWith('image/') ? 'IMAGE' : 'VIDEO'}
            </span>
            <span style={{ color:'var(--text-faint)', fontSize:9 }}>
              {(file.size/1024/1024).toFixed(2)} MB
            </span>
          </div>
        </div>
      )}

      {/* Comment block */}
      <div style={{
        padding:'10px 14px', background:'var(--surface)', border:'1px solid var(--border)',
        borderLeft:'2px solid var(--cyan-dim)',
      }}>
        <p style={{ color:'var(--text-faint)', fontSize:9, lineHeight:1.9, letterSpacing:'0.04em' }}>
          // Analyse des artefacts visuels, cohérence temporelle<br/>
          // et signatures fréquentielles (GradCAM + Temporal)<br/>
          // Modèle : EfficientNet-B0 fine-tuné · seuil 50%
        </p>
      </div>
    </div>
  )
}