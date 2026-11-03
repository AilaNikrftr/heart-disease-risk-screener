# Heart Disease Risk Screener

A full-stack web app that screens adults for coronary heart disease risk from nine routine measurements: age, sex, BMI, blood pressure, cholesterol, smoking and diabetes.

- **Model:** scikit-learn logistic regression trained on the CDC's **NHANES 2015–2016** survey (adults 20+)
- **API:** FastAPI, with screenings stored in SQLite
- **Frontend:** React + TypeScript (Vite)
- **Tooling:** pytest and Vitest tests, GitHub Actions CI, Docker Compose

> Educational project, not a medical device. The score is not a diagnosis.

## How it works

```
React form ──POST /api/screenings──▶ FastAPI ──▶ scikit-learn pipeline ──▶ score + top factors
                                        │
                                        └──▶ SQLite (screening history)
```

1. `ml/fetch_data.py` downloads eight NHANES tables (demographics, body measures, blood pressure, cholesterol, HDL, smoking, diabetes, medical conditions).
2. `ml/features.py` joins them on the participant id (`SEQN`) and recodes survey answers. For example, a diastolic reading of 0 is treated as missing, and "refused" or "don't know" heart disease answers are dropped because they can't be labelled.
3. The label is whether the person was ever told they had **coronary heart disease, angina, or a heart attack** (`MCQ160C/D/E`).
4. `ml/train.py` holds out a stratified 20% test set and trains:
   - **logistic regression** (median imputation → scaling → `class_weight="balanced"`), which is the served model
   - a **random forest** as a baseline, to check how much accuracy the simpler model gives up
5. The API returns a 0–100 score, a low/moderate/high band, and the **three answers that raised the score most**. For logistic regression, each feature adds `coefficient × scaled value` to the log-odds, so the explanation comes straight from the model.

### Design decisions

- **Recall over precision.** Only a small share of adults report heart disease. A screener that misses most of them is useless, so the classes are balanced during training and the model accepts more false positives in exchange.
- **The score is a ranking, not a probability.** Balanced class weights shift the scores upward, so the bands are relative risk levels and the UI doesn't show the score as a percentage chance.
- **Explainable model first.** The random forest is kept only as a baseline. If it clearly outperformed logistic regression, that would be the signal to switch and explain it another way (e.g. SHAP).
- **Validation at the API boundary.** Pydantic rejects impossible values (a BMI of 250, or diastolic pressure above systolic) and unknown fields, before anything reaches the model.

## Results

CI's `train` job trains on the real NHANES data on every push and writes the held-out metrics to the job summary and the `model` artifact (`metrics.json`).

## Run it

### With Docker

```bash
docker compose up --build
```

Open http://localhost:8080. The backend image downloads NHANES and trains the model while it builds.

### Locally

Backend (Python 3.11+):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m ml.fetch_data     # downloads NHANES 2015-2016 into ml/data/
python -m ml.train          # writes ml/artifacts/model.joblib + metrics.json
uvicorn app.main:create_app --factory --reload
```

Frontend (Node 20+), in another terminal:

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173, proxies /api to :8000
```

### Tests

```bash
cd backend && pytest        # features, training, API
cd frontend && npm test     # form, results, error messages
```

The backend tests never download NHANES. They generate small tables with the same columns and codes, where risk factors really do raise heart disease rates, so the tests can check that training finds a signal.

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Whether a model is loaded, plus its metrics |
| `POST` | `/api/screenings` | Score a profile and store it |
| `GET` | `/api/screenings?limit=20` | Recent screenings, newest first |
| `DELETE` | `/api/screenings/{id}` | Delete a screening |

Interactive docs: http://localhost:8000/docs

## Limitations and next steps

- Heart disease is **self-reported** in NHANES, and the data is cross-sectional: people already diagnosed may have changed their habits or started medication, which weakens some associations.
- NHANES uses **survey weights** to represent the US population. This model is trained unweighted, so its metrics describe the sample, not the population.
- Next steps: calibrate the scores (e.g. `CalibratedClassifierCV`) so they can be shown as probabilities, check performance separately for men and women, and add authentication so history is per user instead of shared.

## Data

National Center for Health Statistics, National Health and Nutrition Examination Survey (NHANES) 2015–2016. Public-use data files from the CDC.
