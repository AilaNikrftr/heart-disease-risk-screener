import type { Screening } from '../api'

interface Props {
  items: Screening[]
  onDelete: (id: number) => void
}

export function History({ items, onDelete }: Props) {
  if (items.length === 0) return null
  return (
    <section className="card">
      <h2>Recent screenings</h2>
      <table>
        <thead>
          <tr>
            <th>When</th>
            <th>Age</th>
            <th>Score</th>
            <th>Band</th>
            <th aria-label="Actions" />
          </tr>
        </thead>
        <tbody>
          {items.map((s) => (
            <tr key={s.id}>
              <td>{new Date(s.created_at).toLocaleString()}</td>
              <td>{s.answers.age}</td>
              <td>{Math.round(s.score * 100)}</td>
              <td className={`band-text-${s.band}`}>{s.band}</td>
              <td>
                <button
                  type="button"
                  className="link"
                  aria-label={`Delete screening ${s.id}`}
                  onClick={() => onDelete(s.id)}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
