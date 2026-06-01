import { useState, useEffect, useCallback, useRef } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, ReferenceLine
} from 'recharts'
import axios from 'axios'
import './index.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const AUTO_INTERVAL = 30 // seconds between each auto prediction

// ── Custom Tooltip for chart ──────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'rgba(14,20,32,0.95)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 10, padding: '10px 14px', fontSize: 12
    }}>
      <p style={{ color: '#94a3b8', marginBottom: 4 }}>{label}</p>
      <p style={{ color: '#818cf8', fontFamily: 'JetBrains Mono,monospace', fontWeight: 600 }}>
        ₹{Number(payload[0]?.value).toFixed(2)}
      </p>
    </div>
  )
}

// ── Signal icon ───────────────────────────────────────────────────────────────
function SignalIcon({ signal }) {
  if (signal === 'BUY')  return <span>📈</span>
  if (signal === 'SELL') return <span>📉</span>
  return <span>⏸</span>
}

// ── Confidence bar ─────────────────────────────────────────────────────────────
function ConfidenceBar({ signal, confidence }) {
  const pct = signal === 'SELL'
    ? Math.round((1 - confidence) * 100)
    : Math.round(confidence * 100)

  return (
    <div className="confidence-bar-wrap">
      <div className="confidence-label">
        <span>Model Confidence</span>
        <span className="mono">{pct}%</span>
      </div>
      <div className="confidence-track">
        <div
          className={`confidence-fill ${signal}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ── Auto Mode Status Banner ───────────────────────────────────────────────────
function AutoBanner({ countdown, position, pnl, totalTrades }) {
  return (
    <div className="auto-banner">
      <div className="auto-banner-left">
        <span className="auto-pulse" />
        <span className="auto-label">AUTO MODE ACTIVE</span>
        <span className="auto-countdown mono">Next scan in {countdown}s</span>
      </div>
      <div className="auto-banner-stats">
        <div className="auto-stat">
          <span className="auto-stat-label">Position</span>
          <span className={`auto-stat-value ${position ? 'BUY' : ''}`} style={{ color: position ? 'var(--green)' : 'var(--text-muted)' }}>
            {position ? `LONG @ ₹${position.entryPrice.toFixed(2)}` : 'FLAT'}
          </span>
        </div>
        <div className="auto-stat">
          <span className="auto-stat-label">Session P&L</span>
          <span className="auto-stat-value mono" style={{ color: pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)} pts
          </span>
        </div>
        <div className="auto-stat">
          <span className="auto-stat-label">Trades</span>
          <span className="auto-stat-value mono">{totalTrades}</span>
        </div>
      </div>
    </div>
  )
}

// ── Trade Log ──────────────────────────────────────────────────────────────────
function TradeLog({ log }) {
  return (
    <div className="glass-card trade-log">
      <div className="card-header">
        <span className="card-title">📋 Simulated Trade Log</span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {log.length} entries
        </span>
      </div>
      <div className="log-list">
        {log.length === 0 && (
          <p className="no-data">No trades yet — select a ticker and predict.</p>
        )}
        {[...log].reverse().map((entry, i) => (
          <div className="log-item" key={i}>
            <span className={`log-signal ${entry.signal}`}>{entry.signal}</span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
              {entry.ticker}
            </span>
            <span className="log-price">₹{entry.price}</span>
            {entry.pnl !== undefined && (
              <span className="mono" style={{
                fontSize: '0.75rem', fontWeight: 600,
                color: entry.pnl >= 0 ? 'var(--green)' : 'var(--red)'
              }}>
                {entry.pnl >= 0 ? '+' : ''}{entry.pnl} pts
              </span>
            )}
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
              {entry.conf}% conf.
            </span>
            <span className="log-time">{entry.time}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Setup Page ─────────────────────────────────────────────────────────────────
function SetupPage({ onComplete }) {
  const [source, setSource] = useState('yfinance')
  const [apiKey, setApiKey] = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await axios.post(`${API}/api/config`, {
        data_source: source,
        api_key: apiKey,
        api_secret: apiSecret
      })
      if (res.data.redirect_url) {
        window.location.href = res.data.redirect_url
      } else {
        onComplete()
      }
    } catch (err) {
      alert(err.response?.data?.detail || "Configuration failed")
      setLoading(false)
    }
  }

  return (
    <div className="setup-container">
      <div className="glass-card setup-card" style={{ position: 'relative' }}>
        <button 
          className="btn-close" 
          onClick={onComplete}
          title="Close Settings"
        >
          ×
        </button>
        <h2 style={{ marginBottom: 8 }}>Welcome to StonksAI</h2>
        <p style={{ color: 'var(--text-muted)', marginBottom: 24, fontSize: '0.9rem' }}>
          Select your market data source to continue.
        </p>

        <form onSubmit={handleSubmit} className="setup-form">
          <div className="source-options">
            <label className={`source-option ${source === 'yfinance' ? 'active' : ''}`}>
              <input type="radio" name="source" value="yfinance" checked={source === 'yfinance'} onChange={() => setSource('yfinance')} />
              <div className="source-content">
                <strong>Yahoo Finance (Free)</strong>
                <p>Public data, slightly delayed. Good for testing.</p>
              </div>
            </label>
            <label className={`source-option ${source === 'kite' ? 'active' : ''}`}>
              <input type="radio" name="source" value="kite" checked={source === 'kite'} onChange={() => setSource('kite')} />
              <div className="source-content">
                <strong>Zerodha Kite (Premium)</strong>
                <p>Real-time, minute-level data. Requires API Key.</p>
              </div>
            </label>
          </div>

          {source === 'kite' && (
            <div className="kite-creds slide-down">
              <div className="input-group">
                <label>API Key</label>
                <input type="text" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="Enter your Kite API Key" required />
              </div>
              <div className="input-group">
                <label>API Secret</label>
                <input type="password" value={apiSecret} onChange={e => setApiSecret(e.target.value)} placeholder="Enter your Kite API Secret" required />
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8 }}>
                Get these from <a href="https://developers.kite.trade/" target="_blank" rel="noreferrer" style={{color: 'var(--accent)'}}>developers.kite.trade</a>
              </p>
            </div>
          )}

          <button type="submit" className="btn-setup-submit" disabled={loading}>
            {loading ? 'Saving...' : (source === 'kite' ? 'Save & Login with Zerodha' : 'Continue with YFinance')}
          </button>
        </form>
      </div>
    </div>
  )
}

export default function App() {
  const [tickers, setTickers] = useState([])
  const [watchlist, setWatchlist] = useState([])
  const [newTicker, setNewTicker] = useState('')
  const [activeTicker, setActiveTicker] = useState('')
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [log, setLog]         = useState([])

  // Auto mode state
  const [autoMode, setAutoMode]       = useState(false)
  const [countdown, setCountdown]     = useState(AUTO_INTERVAL)
  const [position, setPosition]       = useState(null)  // { entryPrice, ticker }
  const [sessionPnL, setSessionPnL]   = useState(0)
  const [totalTrades, setTotalTrades] = useState(0)
  const [kiteStatus, setKiteStatus]   = useState(null)
  
  // View state: 'loading' | 'setup' | 'dashboard'
  const [view, setView] = useState('loading')

  const intervalRef  = useRef(null)
  const countdownRef = useRef(null)
  const autoModeRef  = useRef(false)

  // Check Kite status on mount
  useEffect(() => {
    axios.get(`${API}/api/status`)
      .then(r => {
        setKiteStatus(r.data)
        if (r.data.is_configured) {
          setView('dashboard')
        } else {
          setView('setup')
        }
      })
      .catch(() => {
        setView('dashboard') // fallback
      })
      
    // Fetch dynamic tickers list
    axios.get(`${API}/api/tickers`)
      .then(r => {
        setTickers(r.data)
        if (r.data.length > 0) setActiveTicker(r.data[0])
      })
      .catch(console.error)
      
    // Fetch watchlist
    axios.get(`${API}/api/watchlist`)
      .then(r => setWatchlist(r.data))
      .catch(console.error)
  }, [])

  const handleAddWatchlist = async (e) => {
    e.preventDefault()
    if (!newTicker.trim()) return
    try {
      const res = await axios.post(`${API}/api/watchlist/${newTicker.trim()}`)
      setWatchlist(res.data)
      setNewTicker('')
    } catch (err) {
      console.error("Failed to add ticker", err)
    }
  }

  const handleRemoveWatchlist = async (t) => {
    try {
      const res = await axios.delete(`${API}/api/watchlist/${t}`)
      setWatchlist(res.data)
      if (activeTicker === t && tickers.length > 0) setActiveTicker(tickers[0])
    } catch (err) {
      console.error("Failed to remove ticker", err)
    }
  }

  // Keep ref in sync so callbacks always see latest
  useEffect(() => { autoModeRef.current = autoMode }, [autoMode])

  // Process auto-trade logic after receiving a prediction
  const processAutoTrade = useCallback((d) => {
    const price = d.current_price
    const signal = d.signal
    const conf = signal === 'SELL'
      ? Math.round((1 - d.confidence) * 100)
      : Math.round(d.confidence * 100)

    let tradePnL = undefined

    setPosition(prev => {
      if (signal === 'BUY' && !prev) {
        // Enter LONG position
        setTotalTrades(t => t + 1)
        return { entryPrice: price, ticker: d.ticker }
      }
      if (signal === 'SELL' && prev) {
        // Exit position — calculate P&L
        tradePnL = parseFloat((price - prev.entryPrice).toFixed(2))
        setSessionPnL(pnl => parseFloat((pnl + tradePnL).toFixed(2)))
        setTotalTrades(t => t + 1)
        return null
      }
      return prev
    })

    setLog(prev => [...prev, {
      ticker: d.ticker,
      signal: signal === 'BUY' && !position ? 'ENTER' : signal === 'SELL' && position ? 'EXIT' : signal,
      price:  price.toFixed(2),
      conf,
      pnl:    tradePnL,
      time:   new Date().toLocaleTimeString(),
    }])
  }, [position])

  const fetchPrediction = useCallback(async (ticker, isAuto = false) => {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.get(`${API}/api/predict/${ticker}`)
      setData(res.data)

      if (isAuto) {
        processAutoTrade(res.data)
      } else {
        // Manual mode — just log
        const d = res.data
        setLog(prev => [...prev, {
          ticker: d.ticker,
          signal: d.signal,
          price:  d.current_price.toFixed(2),
          conf:   d.signal === 'SELL'
            ? Math.round((1 - d.confidence) * 100)
            : Math.round(d.confidence * 100),
          time:   new Date().toLocaleTimeString(),
        }])
      }
    } catch (e) {
      setError(e?.response?.data?.detail || 'Backend unreachable. Is the FastAPI server running?')
    } finally {
      setLoading(false)
    }
  }, [processAutoTrade])

  // Auto mode: countdown + fetch cycle
  useEffect(() => {
    if (autoMode) {
      // Immediately fetch on start
      fetchPrediction(activeTicker, true)
      setCountdown(AUTO_INTERVAL)

      // Countdown ticker every second
      countdownRef.current = setInterval(() => {
        setCountdown(c => c - 1)
      }, 1000)

      // Fetch every AUTO_INTERVAL seconds
      intervalRef.current = setInterval(() => {
        setCountdown(AUTO_INTERVAL)
        if (autoModeRef.current) {
          fetchPrediction(activeTicker, true)
        }
      }, AUTO_INTERVAL * 1000)

    } else {
      clearInterval(intervalRef.current)
      clearInterval(countdownRef.current)
      setCountdown(AUTO_INTERVAL)
    }
    return () => {
      clearInterval(intervalRef.current)
      clearInterval(countdownRef.current)
    }
  }, [autoMode, activeTicker]) // intentionally not including fetchPrediction to avoid infinite loop

  const toggleAutoMode = () => {
    if (!autoMode) {
      // Starting auto mode — reset session
      setPosition(null)
      setSessionPnL(0)
      setTotalTrades(0)
    }
    setAutoMode(a => !a)
  }

  const handleTicker = (t) => {
    if (autoMode) return // lock ticker during auto mode
    setActiveTicker(t)
    setData(null)
    setError(null)
  }

  // Chart data — raw close prices from candles
  const chartData = data?.candles?.map(c => ({
    time: new Date(c.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    close: c.close,
  })) ?? []

  const minClose = chartData.length ? Math.min(...chartData.map(d => d.close)) : 0
  const maxClose = chartData.length ? Math.max(...chartData.map(d => d.close)) : 0

  // Stats
  const priceDelta = chartData.length > 1
    ? ((chartData.at(-1).close - chartData[0].close) / chartData[0].close * 100).toFixed(2)
    : null

  if (view === 'loading') {
    return (
      <div className="app-shell" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
        <div className="spinner" />
      </div>
    )
  }

  if (view === 'setup') {
    return <SetupPage onComplete={() => setView('dashboard')} />
  }

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">🤖</div>
          <div>
            <h1>StonksAI</h1>
            <p className="header-subtitle">AI-Powered Trading Intelligence</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button className="btn-settings" onClick={() => setView('setup')} title="Settings">⚙️</button>
          {kiteStatus && !kiteStatus.kite_linked && kiteStatus.kite_api_key_set && (
            <a href={`${API}/api/login`} className="btn-zerodha">Login with Zerodha</a>
          )}
          <button
            id="auto-mode-btn"
            className={`btn-auto ${autoMode ? 'active' : ''}`}
            onClick={toggleAutoMode}
          >
            {autoMode ? '⏹ Stop Auto Trading' : '▶ Start Auto Trading'}
          </button>
          <div className="status-dot">
            {kiteStatus?.mode || (autoMode ? 'Auto' : 'Manual')}
          </div>
        </div>
      </header>

      {/* ── Auto Mode Banner ── */}
      {autoMode && (
        <AutoBanner
          countdown={Math.max(0, countdown)}
          position={position}
          pnl={sessionPnL}
          totalTrades={totalTrades}
        />
      )}

      {/* ── Ticker bar ── */}
      <div className="ticker-bar-wrap">
        <div className="ticker-bar">
          {tickers.map(t => (
            <button
              key={t}
              id={`ticker-${t}`}
              className={`ticker-btn ${activeTicker === t ? 'active' : ''} ${autoMode ? 'disabled' : ''}`}
              onClick={() => handleTicker(t)}
              disabled={autoMode}
            >
              {t}
            </button>
          ))}
          {watchlist.filter(w => !tickers.includes(w)).map(t => (
            <div key={t} className={`ticker-btn-group ${activeTicker === t ? 'active' : ''}`}>
              <button
                className={`ticker-btn ${activeTicker === t ? 'active' : ''} ${autoMode ? 'disabled' : ''}`}
                onClick={() => handleTicker(t)}
                disabled={autoMode}
              >
                {t}
              </button>
              <button className="ticker-remove-btn" onClick={() => handleRemoveWatchlist(t)} disabled={autoMode}>×</button>
            </div>
          ))}
        </div>
        <form onSubmit={handleAddWatchlist} className="add-ticker-form">
          <input 
            type="text" 
            placeholder="Add NSE Symbol..." 
            value={newTicker} 
            onChange={e => setNewTicker(e.target.value.toUpperCase())}
            disabled={autoMode}
          />
          <button type="submit" disabled={autoMode || !newTicker.trim()}>+</button>
        </form>
      </div>

      {/* ── Dashboard grid ── */}
      <div className="dashboard-grid">

        {/* ── Chart card ── */}
        <div className="glass-card chart-card">
          <div className="price-hero">
            <div>
              <p className="price-label">Current Price — {activeTicker}</p>
              <p className="price-value">
                <span className="price-currency">₹</span>
                {data ? data.current_price.toLocaleString('en-IN', { minimumFractionDigits: 2 }) : '—'}
              </p>
              {priceDelta !== null && (
                <p style={{
                  fontSize: '0.85rem', fontWeight: 600, marginTop: 4,
                  color: priceDelta >= 0 ? 'var(--green)' : 'var(--red)'
                }}>
                  {priceDelta >= 0 ? '▲' : '▼'} {Math.abs(priceDelta)}% (60 min window)
                </p>
              )}
            </div>
            {!autoMode && (
              <button
                id="predict-btn"
                className="btn-refresh"
                onClick={() => fetchPrediction(activeTicker)}
                disabled={loading}
              >
                {loading
                  ? <><div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> Analyzing…</>
                  : '⚡ Predict Now'
                }
              </button>
            )}
          </div>

          {error && <div className="error-box">⚠️ {error}</div>}

          {loading && !data && (
            <div className="loading-overlay">
              <div className="spinner" />
              <p>Fetching live market data…</p>
            </div>
          )}

          {!loading && !data && !error && (
            <div className="loading-overlay">
              <p style={{ fontSize: '3rem' }}>📊</p>
              <p>Select a ticker and click <strong>Predict Now</strong> or start <strong>Auto Trading</strong>.</p>
            </div>
          )}

          {data && (
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"   stopColor="#6366f1" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
                  <XAxis
                    dataKey="time"
                    tick={{ fill: '#475569', fontSize: 10 }}
                    axisLine={false} tickLine={false}
                    interval={Math.floor(chartData.length / 6)}
                  />
                  <YAxis
                    domain={[minClose * 0.9995, maxClose * 1.0005]}
                    tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                    axisLine={false} tickLine={false}
                    tickFormatter={v => `₹${v.toFixed(0)}`}
                    width={64}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine
                    y={data.current_price}
                    stroke="rgba(99,102,241,0.5)"
                    strokeDasharray="4 4"
                  />
                  <Area
                    type="monotone"
                    dataKey="close"
                    stroke="#6366f1"
                    strokeWidth={2}
                    fill="url(#priceGrad)"
                    dot={false}
                    activeDot={{ r: 4, fill: '#818cf8', strokeWidth: 0 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* ── Stats row ── */}
        <div className="stats-row">
          {[
            { label: '60-min High', value: data ? `₹${maxClose.toFixed(2)}` : '—', color: 'var(--green)' },
            { label: '60-min Low',  value: data ? `₹${minClose.toFixed(2)}` : '—', color: 'var(--red)'   },
            { label: 'Candles',     value: data ? data.candles.length : '—',        color: 'var(--accent-bright)' },
            { label: 'Horizon',     value: '15 min',                                color: 'var(--amber)'  },
          ].map((s, i) => (
            <div className="stat-card" key={i}>
              <p className="stat-label">{s.label}</p>
              <p className="stat-value" style={{ color: s.color }}>{s.value}</p>
            </div>
          ))}
        </div>

        {/* ── Prediction Panel ── */}
        <div className="prediction-col">
          <div className="glass-card">
            <p className="card-title" style={{ marginBottom: 16 }}>🤖 AI Signal</p>

            {!data && !loading && (
              <p className="no-data">Run a prediction to see the AI signal.</p>
            )}

            {loading && (
              <div className="loading-overlay" style={{ padding: 32 }}>
                <div className="spinner" />
              </div>
            )}

            {data && !loading && (
              <div className="prediction-panel">
                <div className={`signal-badge ${data.signal}`}>
                  <SignalIcon signal={data.signal} />
                  {data.signal}
                </div>
                <ConfidenceBar signal={data.signal} confidence={data.confidence} />
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  The model predicts the price of <strong style={{ color: 'var(--text-primary)' }}>{data.ticker}</strong> will&nbsp;
                  {data.signal === 'BUY'  && <span style={{ color: 'var(--green)'  }}>rise</span>}
                  {data.signal === 'SELL' && <span style={{ color: 'var(--red)'    }}>fall</span>}
                  {data.signal === 'HOLD' && <span style={{ color: 'var(--amber)'  }}>stay flat</span>}
                  &nbsp;over the next <strong style={{ color: 'var(--text-primary)' }}>15 minutes</strong>.
                </div>
              </div>
            )}
          </div>

          {/* ── Model info card ── */}
          <div className="glass-card">
            <p className="card-title" style={{ marginBottom: 14 }}>🧠 Model Info</p>
            {[
              ['Architecture', 'LSTM (3 layers)'],
              ['Hidden Size', '128 units'],
              ['Input Window', '60 minutes'],
              ['Prediction', '15 min horizon'],
              ['Training Data', '5 stocks × 20k min'],
              ['Normalization', 'Rolling Z-Score'],
            ].map(([k, v]) => (
              <div key={k} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '8px 0', borderBottom: '1px solid var(--border)',
                fontSize: '0.8rem'
              }}>
                <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                <span className="mono" style={{ color: 'var(--accent-bright)', fontSize: '0.78rem' }}>{v}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Trade Log ── */}
        <TradeLog log={log} />
      </div>
    </div>
  )
}
