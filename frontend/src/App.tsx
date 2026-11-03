import { useCallback, useEffect, useState } from 'react'
import { api, type Health, type Screening, type ScreeningInput } from './api'
import { History } from './components/History'
import { ResultCard } from './components/ResultCard'
import { ScreeningForm } from './components/ScreeningForm'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [result, setResult] = useState<Screening | null>(null)
  const [history, setHistory] = useState<Screening[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const loadHistory = useCallback(async () => {
    try {
      setHistory(await api.history())
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  useEffect(() => {
    api.health().then(setHealth, () => setError('Cannot reach the API. Is the backend running?'))
    api.history().then(setHistory, (e: Error) => setError(e.message))
  }, [])

  async function handleSubmit(input: ScreeningInput) {
    setBusy(true)
    setError(null)
    try {
      setResult(await api.screen(input))
      await loadHistory()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function handleDelete(id: number) {
    try {
      await api.remove(id)
      if (result?.id === id) setResult(null)
      await loadHistory()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const metrics = health?.metrics?.logistic_regression

  return (
    <main>
      <header>
        <h1>Heart Disease Risk Screener</h1>
        <p>
          Enter a few routine health measurements to see how your profile compares with adults in
          the CDC's NHANES 2015–2016 survey who reported coronary heart disease, angina, or a heart
          attack.
        </p>
        {health && !health.model_loaded && (
          <p className="warning">The model has not been trained yet. See the README.</p>
        )}
      </header>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      <ScreeningForm onSubmit={handleSubmit} busy={busy} />
      {result && <ResultCard result={result} />}
      <History items={history} onDelete={handleDelete} />

      {metrics && (
        <footer>
          Model: logistic regression trained on {health?.metrics?.rows.toLocaleString()} adults ·
          held-out recall {Math.round(metrics.recall * 100)}% · ROC AUC {metrics.roc_auc}
        </footer>
      )}
    </main>
  )
}
