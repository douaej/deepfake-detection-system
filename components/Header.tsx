export default function Header() {
  return (
    <header style={{
      borderBottom: '1px solid var(--border)',
      background: 'linear-gradient(180deg, rgba(0,229,255,0.025) 0%, transparent 100%)',
    }}>
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '20px 28px' }}>

        {/* Main row */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>

          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            {/* Logo */}
            <div style={{
              width: 44, height: 44, position: 'relative', flexShrink: 0,
              border: '1px solid var(--cyan)', background: 'var(--cyan-glow)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 24px rgba(0,229,255,0.15)',
            }}>
              <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                <circle cx="9" cy="9" r="6" stroke="#00e5ff" strokeWidth="1.5"/>
                <line x1="13.5" y1="13.5" x2="20" y2="20" stroke="#00e5ff" strokeWidth="1.5" strokeLinecap="round"/>
                <circle cx="9" cy="9" r="2.5" fill="#00e5ff" opacity="0.4"/>
              </svg>
              {/* Corner ticks */}
              {[
                { top:-1, left:-1, borderTop:'2px solid var(--cyan)', borderLeft:'2px solid var(--cyan)' },
                { top:-1, right:-1, borderTop:'2px solid var(--cyan)', borderRight:'2px solid var(--cyan)' },
                { bottom:-1, left:-1, borderBottom:'2px solid var(--cyan)', borderLeft:'2px solid var(--cyan)' },
                { bottom:-1, right:-1, borderBottom:'2px solid var(--cyan)', borderRight:'2px solid var(--cyan)' },
              ].map((s, i) => (
                <div key={i} style={{ position:'absolute', width:8, height:8, ...s }}/>
              ))}
            </div>

            <div>
              <h1 style={{
                fontFamily: 'var(--display)', fontSize: 20, fontWeight: 800,
                letterSpacing: '-0.02em', color: '#fff', lineHeight: 1.1,
              }}>
                DEEPFAKE<span style={{ color: 'var(--cyan)' }}>_</span>DETECTOR
              </h1>
              <p style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.18em', marginTop: 3 }}>
                FORENSIC ANALYSIS SYSTEM · v1.1
              </p>
            </div>
          </div>

          {/* Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: 7, height: 7, borderRadius: '50%',
              background: 'var(--green)', color: 'var(--green)',
            }} className="pulse-dot"/>
            <span style={{ fontFamily: 'var(--mono)', color: 'var(--text-dim)', fontSize: 9, letterSpacing: '0.14em' }}>
              SYSTEM ONLINE
            </span>
          </div>
        </div>

        {/* Meta bar */}
        <div style={{
          display: 'flex', gap: 28, flexWrap: 'wrap',
          paddingTop: 12, borderTop: '1px solid var(--border)',
        }}>
          {[
            ['MODEL', 'EfficientNet-B0'],
            ['BACKEND', 'PyTorch · CPU'],
            ['XAI', 'GradCAM + Temporel'],
            ['ACCURACY', '~85% img'],
          ].map(([label, value]) => (
            <div key={label} style={{ display: 'flex', gap: 7, alignItems: 'center' }}>
              <span style={{ color: 'var(--text-faint)', fontSize: 9, letterSpacing: '0.14em' }}>{label}</span>
              <span style={{ color: 'var(--cyan-dim)', fontSize: 9 }}>{value}</span>
            </div>
          ))}
        </div>
      </div>
    </header>
  )
}