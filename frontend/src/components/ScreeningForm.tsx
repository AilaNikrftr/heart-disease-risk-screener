import { useState, type FormEvent } from 'react'
import type { ScreeningInput, Sex } from '../api'

interface NumberField {
  key: keyof Pick<
    ScreeningInput,
    'age' | 'bmi' | 'systolic_bp' | 'diastolic_bp' | 'total_cholesterol' | 'hdl_cholesterol'
  >
  label: string
  unit?: string
  min: number
  max: number
  step?: number
}

const NUMBER_FIELDS: NumberField[] = [
  { key: 'age', label: 'Age', unit: 'years', min: 20, max: 100 },
  { key: 'bmi', label: 'BMI', min: 12, max: 80, step: 0.1 },
  { key: 'systolic_bp', label: 'Systolic blood pressure', unit: 'mmHg', min: 70, max: 260 },
  { key: 'diastolic_bp', label: 'Diastolic blood pressure', unit: 'mmHg', min: 30, max: 160 },
  { key: 'total_cholesterol', label: 'Total cholesterol', unit: 'mg/dL', min: 70, max: 500 },
  { key: 'hdl_cholesterol', label: 'HDL cholesterol', unit: 'mg/dL', min: 10, max: 200 },
]

type Draft = Record<NumberField['key'], string> & {
  sex: Sex
  current_smoker: boolean
  diabetes: boolean
}

const EMPTY: Draft = {
  age: '',
  bmi: '',
  systolic_bp: '',
  diastolic_bp: '',
  total_cholesterol: '',
  hdl_cholesterol: '',
  sex: 'female',
  current_smoker: false,
  diabetes: false,
}

interface Props {
  onSubmit: (input: ScreeningInput) => void
  busy: boolean
}

export function ScreeningForm({ onSubmit, busy }: Props) {
  const [draft, setDraft] = useState<Draft>(EMPTY)

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    onSubmit({
      age: Number(draft.age),
      bmi: Number(draft.bmi),
      systolic_bp: Number(draft.systolic_bp),
      diastolic_bp: Number(draft.diastolic_bp),
      total_cholesterol: Number(draft.total_cholesterol),
      hdl_cholesterol: Number(draft.hdl_cholesterol),
      sex: draft.sex,
      current_smoker: draft.current_smoker,
      diabetes: draft.diabetes,
    })
  }

  return (
    <form className="card form" onSubmit={handleSubmit}>
      <div className="grid">
        {NUMBER_FIELDS.map((f) => (
          <label key={f.key}>
            <span>
              {f.label}
              {f.unit && <small> ({f.unit})</small>}
            </span>
            <input
              type="number"
              name={f.key}
              required
              min={f.min}
              max={f.max}
              step={f.step ?? 1}
              value={draft[f.key]}
              onChange={(e) => setDraft({ ...draft, [f.key]: e.target.value })}
            />
          </label>
        ))}
        <label>
          <span>Sex</span>
          <select
            name="sex"
            value={draft.sex}
            onChange={(e) => setDraft({ ...draft, sex: e.target.value as Sex })}
          >
            <option value="female">Female</option>
            <option value="male">Male</option>
          </select>
        </label>
      </div>
      <div className="checks">
        <label className="check">
          <input
            type="checkbox"
            checked={draft.current_smoker}
            onChange={(e) => setDraft({ ...draft, current_smoker: e.target.checked })}
          />
          I currently smoke
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={draft.diabetes}
            onChange={(e) => setDraft({ ...draft, diabetes: e.target.checked })}
          />
          A doctor has told me I have diabetes
        </label>
      </div>
      <button type="submit" disabled={busy}>
        {busy ? 'Scoring…' : 'Check my risk'}
      </button>
    </form>
  )
}
