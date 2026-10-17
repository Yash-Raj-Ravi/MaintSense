import { useEffect, useState, type FormEvent } from 'react'
import {
  ApiError,
  getHealth,
  getModelInfo,
  NetworkError,
  predict,
  type HealthResponse,
  type MachineType,
  type ModelInfoResponse,
  type PredictionRequest,
  type PredictionResponse,
} from './api'

type FormValues = {
  Type: MachineType
  'Air temperature [K]': string
  'Process temperature [K]': string
  'Rotational speed [rpm]': string
  'Torque [Nm]': string
  'Tool wear [min]': string
}

type ConnectionState = 'checking' | 'online' | 'degraded' | 'offline'

const initialValues: FormValues = {
  Type: 'M',
  'Air temperature [K]': '300',
  'Process temperature [K]': '310',
  'Rotational speed [rpm]': '1500',
  'Torque [Nm]': '40',
  'Tool wear [min]': '120',
}

const features: { key: Exclude<keyof FormValues, 'Type'>; label: string; unit: string }[] = [
  { key: 'Air temperature [K]', label: 'Air temperature', unit: 'K' },
  { key: 'Process temperature [K]', label: 'Process temperature', unit: 'K' },
  { key: 'Rotational speed [rpm]', label: 'Rotational speed', unit: 'rpm' },
  { key: 'Torque [Nm]', label: 'Torque', unit: 'Nm' },
  { key: 'Tool wear [min]', label: 'Tool wear', unit: 'min' },
]

const evaluation = {
  validation: {
    precision: 0.8824,
    recall: 0.6618,
    f1: 0.7563,
    averagePrecision: 0.7772,
    accuracy: 0.9855,
    falsePositives: 6,
    falseNegatives: 23,
    matrix: { tn: 1926, fp: 6, fn: 23, tp: 45 },
  },
  test: {
    precision: 0.8367,
    recall: 0.6029,
    f1: 0.7009,
    averagePrecision: 0.7817,
    accuracy: 0.9825,
    falsePositives: 8,
    falseNegatives: 27,
    matrix: { tn: 1924, fp: 8, fn: 27, tp: 41 },
  },
}

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    grid: 'M3 3h7v7H3z M14 3h7v7h-7z M14 14h7v7h-7z M3 14h7v7H3z',
    gauge: 'M20.8 14a9 9 0 1 0-17.6 0 M12 12l4-4 M5 18h14',
    chart: 'M4 19V5 M4 19h17 M8 15l3-4 3 2 5-7',
    info: 'M12 17v-5 M12 8h.01 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0z',
    arrow: 'M7 17 17 7 M7 7h10v10',
    refresh: 'M20 7v5h-5 M4 17v-5h5 M5.6 9A7 7 0 0 1 18 6l2 2 M4 16l2 2a7 7 0 0 0 12.4-3',
    activity: 'M3 12h4l3-8 4 16 3-8h4',
    shield: 'M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z M9 12l2 2 4-4',
  }
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d={paths[name] ?? paths.activity} />
    </svg>
  )
}

function StatusPill({ state }: { state: ConnectionState }) {
  const label = {
    checking: 'Checking API',
    online: 'API connected',
    degraded: 'Model unavailable',
    offline: 'API offline',
  }[state]
  return <span className={`status-pill status-${state}`}><span className="status-dot" />{label}</span>
}

function MetricBars({ values }: { values: typeof evaluation.validation }) {
  const metrics = [
    ['Precision', values.precision],
    ['Recall', values.recall],
    ['F1 score', values.f1],
    ['Average precision', values.averagePrecision],
  ] as const
  return (
    <div className="metric-bars" aria-label="Model evaluation metrics">
      {metrics.map(([label, value]) => (
        <div className="metric-bar-row" key={label}>
          <span>{label}</span>
          <div className="metric-track" aria-hidden="true"><span style={{ width: `${value * 100}%` }} /></div>
          <strong>{(value * 100).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  )
}

function ConfusionMatrix({ matrix }: { matrix: typeof evaluation.validation.matrix }) {
  return (
    <div className="matrix-wrap">
      <div className="matrix-axis">Predicted class</div>
      <div className="matrix-grid" role="img" aria-label={`Confusion matrix: true negative ${matrix.tn}, false positive ${matrix.fp}, false negative ${matrix.fn}, true positive ${matrix.tp}`}>
        <span className="matrix-corner">Actual</span><span className="matrix-col">0</span><span className="matrix-col">1</span>
        <span className="matrix-row">0</span><span className="matrix-cell matrix-tn"><small>TN</small><strong>{matrix.tn}</strong></span><span className="matrix-cell matrix-fp"><small>FP</small><strong>{matrix.fp}</strong></span>
        <span className="matrix-row">1</span><span className="matrix-cell matrix-fn"><small>FN</small><strong>{matrix.fn}</strong></span><span className="matrix-cell matrix-tp"><small>TP</small><strong>{matrix.tp}</strong></span>
      </div>
      <p className="matrix-note">Rows are actual labels; columns are predicted labels.</p>
    </div>
  )
}

function MetricCard({ label, value, suffix = '%' }: { label: string; value: number; suffix?: string }) {
  return <div className="metric-card"><span>{label}</span><strong>{suffix === '%' ? `${(value * 100).toFixed(2)}%` : value}</strong></div>
}

export default function App() {
  const [connection, setConnection] = useState<ConnectionState>('checking')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [overviewError, setOverviewError] = useState('')
  const [form, setForm] = useState<FormValues>(initialValues)
  const [formError, setFormError] = useState('')
  const [apiError, setApiError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)

  useEffect(() => {
    let active = true
    Promise.all([getHealth(), getModelInfo()])
      .then(([healthResult, infoResult]) => {
        if (!active) return
        setHealth(healthResult)
        setModelInfo(infoResult)
        setConnection(healthResult.model_available ? 'online' : 'degraded')
        if (!healthResult.model_available) setOverviewError('The API is responding, but its model artifact is unavailable.')
      })
      .catch((error: unknown) => {
        if (!active) return
        setConnection('offline')
        setOverviewError(error instanceof Error ? error.message : 'Could not load API status.')
      })
    return () => { active = false }
  }, [])

  function updateField<K extends keyof FormValues>(key: K, value: FormValues[K]) {
    setForm((current) => ({ ...current, [key]: value }))
    setFormError('')
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setApiError('')
    const values: Record<string, number> = {}
    for (const feature of features) {
      const raw = form[feature.key].trim()
      const numeric = Number(raw)
      if (!raw || !Number.isFinite(numeric)) {
        setFormError(`${feature.label} must be a finite number.`)
        return
      }
      values[feature.key] = numeric
    }
    setFormError('')
    setPrediction(null)
    setIsSubmitting(true)
    const payload: PredictionRequest = {
      Type: form.Type,
      'Air temperature [K]': values['Air temperature [K]'],
      'Process temperature [K]': values['Process temperature [K]'],
      'Rotational speed [rpm]': values['Rotational speed [rpm]'],
      'Torque [Nm]': values['Torque [Nm]'],
      'Tool wear [min]': values['Tool wear [min]'],
    }
    try {
      setPrediction(await predict(payload))
    } catch (error) {
      if (error instanceof ApiError) setApiError(`API error (${error.status}): ${error.message}`)
      else if (error instanceof NetworkError) setApiError(error.message)
      else setApiError('The prediction request could not be completed. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const threshold = prediction?.threshold ?? modelInfo?.threshold ?? 0.4

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="MaintSense overview">
          <span className="brand-mark"><Icon name="activity" size={21} /></span>
          <span><strong>MaintSense</strong><small>RELIABILITY CONSOLE</small></span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Dashboard sections">
          <a className="nav-item active" href="#overview"><Icon name="grid" />Overview</a>
          <a className="nav-item" href="#assessment"><Icon name="gauge" />Machine Assessment</a>
          <a className="nav-item" href="#results"><Icon name="activity" />Prediction Results</a>
          <a className="nav-item" href="#performance"><Icon name="chart" />Model Performance</a>
          <a className="nav-item" href="#about"><Icon name="info" />About the Model</a>
        </nav>
        <div className="sidebar-bottom">
          <div className="scope-card"><Icon name="shield" size={16} /><p>Synthetic dataset<br /><strong>Research prototype</strong></p></div>
          <span className="sidebar-version">MAINTSENSE · 0.1.0</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><b>/</b><strong>Overview</strong></div>
          <div className="topbar-right"><StatusPill state={connection} /><span className="topbar-divider" /><span className="environment-label">LOCAL ENVIRONMENT</span></div>
        </header>

        <div className="page-content">
          <section id="overview" className="page-heading">
            <div><p className="eyebrow">MACHINE RELIABILITY · MODEL CONSOLE</p><h1>Overview</h1><p className="heading-subtitle">A transparent view of the MaintSense failure-classification pipeline.</p></div>
            <div className="threshold-chip"><span className="chip-dot" /><span>EXPERIMENTAL THRESHOLD</span><strong>{(threshold * 100).toFixed(0)}%</strong></div>
          </section>

          <section className="overview-grid" aria-label="Model overview">
            <article className="overview-card api-card">
              <div className="card-topline"><span className="card-icon"><Icon name="activity" /></span><span className={`tiny-status tiny-${connection}`} /></div>
              <p className="card-label">API CONNECTION</p><h2>{connection === 'online' ? 'Connected' : connection === 'checking' ? 'Checking…' : connection === 'degraded' ? 'Model unavailable' : 'Not connected'}</h2>
              <p className="card-foot">{health?.status ?? (overviewError ? 'Check backend and URL settings' : 'Waiting for backend response')}</p>
            </article>
            <article className="overview-card">
              <div className="card-topline"><span className="card-icon"><Icon name="chart" /></span><span className="card-kicker">CLASSIFIER</span></div>
              <p className="card-label">MODEL TYPE</p><h2>{modelInfo?.model_type ?? '—'}</h2><p className="card-foot">Persisted scikit-learn pipeline</p>
            </article>
            <article className="overview-card threshold-card">
              <div className="card-topline"><span className="card-icon"><Icon name="gauge" /></span><span className="card-kicker">CONFIGURED</span></div>
              <p className="card-label">DECISION THRESHOLD</p><h2>{modelInfo ? `${(modelInfo.threshold * 100).toFixed(0)}%` : '—'}</h2><p className="card-foot">Experimental setting · not production-approved</p>
            </article>
            <article className="overview-card">
              <div className="card-topline"><span className="card-icon"><Icon name="shield" /></span><span className="card-kicker">INPUT CONTRACT</span></div>
              <p className="card-label">REQUIRED FEATURES</p><h2>{modelInfo?.required_features.length ?? '—'} <small>signals</small></h2><p className="card-foot">Operating measurements + machine type</p>
            </article>
          </section>
          {overviewError && <div className={`notice ${connection === 'offline' ? 'notice-error' : ''}`} role="status">{overviewError}</div>}

          <section id="assessment" className="section-block">
            <div className="section-heading"><div><p className="eyebrow">INFERENCE WORKSPACE</p><h2>Machine Assessment</h2><p>Enter a single machine record to request a prediction from the API.</p></div><span className="section-index">01 / INPUT</span></div>
            <div className="assessment-layout">
              <form className="assessment-card" onSubmit={handleSubmit} noValidate>
                <div className="form-card-heading"><span className="form-heading-icon"><Icon name="gauge" /></span><div><h3>Operating conditions</h3><p>All fields are sent to the configured prediction endpoint.</p></div></div>
                <label className="field type-field"><span>Machine type <em>Required</em></span><select value={form.Type} onChange={(event) => updateField('Type', event.target.value as MachineType)}><option value="L">L · Type L</option><option value="M">M · Type M</option><option value="H">H · Type H</option></select><small>Choose the category recorded for this machine.</small></label>
                <div className="field-grid">
                  {features.map((feature) => <label className="field" key={feature.key}><span>{feature.label} <em>Required</em></span><div className="input-with-unit"><input type="number" inputMode="decimal" step="any" required aria-label={feature.label} value={form[feature.key]} onChange={(event) => updateField(feature.key, event.target.value)} /><span>{feature.unit}</span></div></label>)}
                </div>
                {formError && <p className="form-message error-message" role="alert">{formError}</p>}
                {apiError && <p className="form-message error-message" role="alert">{apiError}</p>}
                <div className="form-footer"><span><span className="lock-dot" />Sent to your configured MaintSense API</span><button className="primary-button" type="submit" disabled={isSubmitting}>{isSubmitting ? <><span className="spinner" />Assessing…</> : <>Run assessment <Icon name="arrow" size={16} /></>}</button></div>
              </form>

              <section id="results" className="result-card" aria-live="polite">
                <div className="result-head"><div><p className="eyebrow">MODEL RESPONSE</p><h3>Prediction Results</h3></div><span className="result-badge">{prediction ? 'LATEST' : 'AWAITING INPUT'}</span></div>
                {isSubmitting ? <div className="result-empty"><span className="large-spinner" /><strong>Running assessment</strong><p>Waiting for the prediction API response…</p></div> : prediction ? <div className="result-content"><div className={`prediction-state ${prediction.predicted_class === 1 ? 'prediction-positive' : 'prediction-negative'}`}><span className="prediction-state-icon"><Icon name={prediction.predicted_class === 1 ? 'activity' : 'shield'} size={19} /></span><span><small>PREDICTED CLASS</small><strong>{prediction.predicted_class === 1 ? 'Failure class' : 'Non-failure class'}</strong></span><b>{prediction.predicted_class}</b></div><div className="probability-block"><div><span>Failure probability</span><strong>{(prediction.failure_probability * 100).toFixed(1)}<small>%</small></strong></div><div className="probability-track"><span style={{ width: `${Math.min(100, Math.max(0, prediction.failure_probability * 100))}%` }} /></div></div><div className="result-threshold"><span>Configured experimental threshold</span><strong>{(prediction.threshold * 100).toFixed(0)}%</strong></div><p className="result-caveat">Classification output from the synthetic-data model. Not an operational alert.</p></div> : <div className="result-empty"><span className="empty-ring"><Icon name="activity" size={25} /></span><strong>No assessment yet</strong><p>Submit the machine conditions to see the API response here.</p></div>}
              </section>
            </div>
          </section>

          <section id="performance" className="section-block">
            <div className="section-heading"><div><p className="eyebrow">DOCUMENTED EVALUATION</p><h2>Model Performance</h2><p>Frozen Random Forest configuration at the configured experimental threshold of 0.40.</p></div><span className="section-index">02 / EVIDENCE</span></div>
            <div className="performance-grid">
              <article className="performance-card">
                <div className="performance-card-head"><div><span className="dataset-tag validation-tag">VALIDATION</span><h3>Selection partition</h3></div><span className="sample-count">2,000 records</span></div>
                <MetricBars values={evaluation.validation} />
                <div className="compact-metrics"><MetricCard label="Accuracy" value={evaluation.validation.accuracy} /><MetricCard label="False positives" value={evaluation.validation.falsePositives} suffix="" /><MetricCard label="False negatives" value={evaluation.validation.falseNegatives} suffix="" /></div>
                <ConfusionMatrix matrix={evaluation.validation.matrix} />
              </article>
              <article className="performance-card test-card">
                <div className="performance-card-head"><div><span className="dataset-tag test-tag">FINAL TEST</span><h3>One-time final estimate</h3></div><span className="sample-count">2,000 records</span></div>
                <MetricBars values={evaluation.test} />
                <div className="compact-metrics"><MetricCard label="Accuracy" value={evaluation.test.accuracy} /><MetricCard label="False positives" value={evaluation.test.falsePositives} suffix="" /><MetricCard label="False negatives" value={evaluation.test.falseNegatives} suffix="" /></div>
                <ConfusionMatrix matrix={evaluation.test.matrix} />
              </article>
            </div>
            <p className="performance-note"><Icon name="info" size={16} />Validation results informed model and threshold selection. The final test partition was evaluated once after selection was frozen; its results were not used to tune the model. Metrics describe this synthetic dataset split only.</p>
          </section>

          <section id="about" className="section-block about-section">
            <div className="section-heading"><div><p className="eyebrow">SCOPE & LIMITATIONS</p><h2>About the Model</h2><p>Understand what this prototype does—and what it does not establish.</p></div><span className="section-index">03 / CONTEXT</span></div>
            <div className="about-grid">
              <article className="about-card"><span className="about-icon"><Icon name="activity" /></span><div><h3>Machine-failure classification</h3><p>A persisted Random Forest pipeline classifies records using machine type and five operating measurements. The browser sends inputs to FastAPI; it does not calculate predictions.</p></div></article>
              <article className="about-card"><span className="about-icon warning-icon"><Icon name="shield" /></span><div><h3>Synthetic dataset only</h3><p>The AI4I 2020 dataset contains simulated industrial operating data. This model has not been validated as a real-world early-warning system.</p></div></article>
              <article className="about-card"><span className="about-icon threshold-icon"><Icon name="gauge" /></span><div><h3>Experimental threshold · 0.40</h3><p>This is the frozen benchmark setting selected using validation data. It is not production-approved and does not encode an agreed cost for false alarms or missed failures.</p></div></article>
            </div>
          </section>
          <footer className="page-footer"><span>MAINTSENSE · MACHINE-FAILURE CLASSIFICATION</span><span>Synthetic-data research prototype <i /> No real-world early-warning validation</span></footer>
        </div>
      </main>
    </div>
  )
}
