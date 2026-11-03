import type { Screening } from '../api'

const BAND_TEXT: Record<Screening['band'], string> = {
  low: 'Lower risk',
  moderate: 'Moderate risk',
  high: 'Higher risk',
}

export function ResultCard({ result }: { result: Screening }) {
  return (
    <section className={`card result band-${result.band}`} aria-live="polite">
      <h2>{BAND_TEXT[result.band]}</h2>
      <p className="score">
        Risk score <strong>{Math.round(result.score * 100)}</strong> / 100
      </p>
      {result.top_factors.length > 0 ? (
        <>
          <h3>What raised your score most</h3>
          <ul>
            {result.top_factors.map((f) => (
              <li key={f.feature}>{f.label}</li>
            ))}
          </ul>
        </>
      ) : (
        <p>None of your answers pushed the score up.</p>
      )}
      <p className="disclaimer">{result.disclaimer}</p>
    </section>
  )
}
