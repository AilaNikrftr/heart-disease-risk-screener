import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from './App'
import { errorMessage, type Screening } from './api'

const result: Screening = {
  id: 1,
  created_at: '2026-11-02T12:00:00Z',
  answers: {
    age: 74,
    sex: 'male',
    bmi: 33,
    systolic_bp: 165,
    diastolic_bp: 92,
    total_cholesterol: 250,
    hdl_cholesterol: 35,
    current_smoker: true,
    diabetes: true,
  },
  score: 0.91,
  band: 'high',
  flagged: true,
  top_factors: [
    { feature: 'age', label: 'Age', contribution: 1.4 },
    { feature: 'diabetes', label: 'Diabetes', contribution: 0.6 },
  ],
  disclaimer: 'Educational screening tool, not a diagnosis.',
}

function json(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status }))
}

function mockApi(postResponse: () => Promise<Response>) {
  let history: Screening[] = []
  return vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
    const url = String(input)
    if (url === '/api/health') {
      return json({
        status: 'ok',
        model_loaded: true,
        metrics: { rows: 5000, logistic_regression: { roc_auc: 0.8, recall: 0.76, precision: 0.2 } },
      })
    }
    if (url.startsWith('/api/screenings') && init?.method === 'POST') {
      return postResponse().then((r) => {
        if (r.ok) history = [result]
        return r
      })
    }
    return json(history)
  })
}

async function fillForm() {
  const user = userEvent.setup()
  const values: Record<string, string> = {
    age: '74',
    bmi: '33',
    systolic_bp: '165',
    diastolic_bp: '92',
    total_cholesterol: '250',
    hdl_cholesterol: '35',
  }
  for (const [name, value] of Object.entries(values)) {
    await user.type(document.querySelector(`input[name="${name}"]`)!, value)
  }
  await user.selectOptions(screen.getByRole('combobox'), 'male')
  await user.click(screen.getByLabelText('I currently smoke'))
  await user.click(screen.getByLabelText(/diabetes/))
  await user.click(screen.getByRole('button', { name: 'Check my risk' }))
}

describe('App', () => {
  it('shows model metrics from the API', async () => {
    mockApi(() => json(result, 201))
    render(<App />)
    expect(await screen.findByText(/recall 76%/)).toBeInTheDocument()
  })

  it('submits the form and shows the result and history', async () => {
    const fetchMock = mockApi(() => json(result, 201))
    render(<App />)
    await fillForm()

    const heading = await screen.findByRole('heading', { name: 'Higher risk' })
    const card = within(heading.closest('section')!)
    expect(card.getByText('91')).toBeInTheDocument()
    expect(card.getByText('Diabetes')).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Recent screenings' })).toBeInTheDocument()

    const post = fetchMock.mock.calls.find(([, init]) => init?.method === 'POST')!
    expect(JSON.parse(post[1]!.body as string)).toEqual(result.answers)
  })

  it('shows validation errors from the API', async () => {
    mockApi(() =>
      json(
        {
          detail: [
            { loc: ['body'], msg: 'Value error, diastolic blood pressure must be lower than systolic' },
          ],
        },
        422,
      ),
    )
    render(<App />)
    await fillForm()
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(
        'diastolic blood pressure must be lower than systolic',
      ),
    )
  })
})

describe('errorMessage', () => {
  it('uses a plain string detail', () => {
    expect(errorMessage(503, { detail: 'Model not trained yet.' })).toBe('Model not trained yet.')
  })

  it('names the field for field errors', () => {
    const body = { detail: [{ loc: ['body', 'bmi'], msg: 'Input should be less than or equal to 80' }] }
    expect(errorMessage(422, body)).toBe('bmi: Input should be less than or equal to 80')
  })

  it('falls back to the status code', () => {
    expect(errorMessage(500, null)).toBe('Request failed (500)')
  })
})
