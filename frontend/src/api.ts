export type Sex = 'male' | 'female'

export interface ScreeningInput {
  age: number
  sex: Sex
  bmi: number
  systolic_bp: number
  diastolic_bp: number
  total_cholesterol: number
  hdl_cholesterol: number
  current_smoker: boolean
  diabetes: boolean
}

export interface Factor {
  feature: string
  label: string
  contribution: number
}

export interface Screening {
  id: number
  created_at: string
  answers: ScreeningInput
  score: number
  band: 'low' | 'moderate' | 'high'
  flagged: boolean
  top_factors: Factor[]
  disclaimer: string
}

export interface ModelMetrics {
  rows: number
  logistic_regression: { roc_auc: number; recall: number; precision: number }
}

export interface Health {
  status: string
  model_loaded: boolean
  metrics: ModelMetrics | null
}

/** Turn FastAPI's error shapes into one readable sentence. */
export function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => {
        const field = d.loc?.filter((p) => p !== 'body').join('.')
        const msg = (d.msg ?? 'Invalid value').replace(/^Value error, /, '')
        return field ? `${field}: ${msg}` : msg
      })
      .join('; ')
  }
  return `Request failed (${status})`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (response.status === 204) return undefined as T
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new Error(errorMessage(response.status, body))
  return body as T
}

export const api = {
  health: () => request<Health>('/api/health'),
  screen: (input: ScreeningInput) =>
    request<Screening>('/api/screenings', { method: 'POST', body: JSON.stringify(input) }),
  history: () => request<Screening[]>('/api/screenings?limit=10'),
  remove: (id: number) => request<void>(`/api/screenings/${id}`, { method: 'DELETE' }),
}
