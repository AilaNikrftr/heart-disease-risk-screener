FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY ml ./ml

# Download NHANES 2015-2016 from the CDC and train the model while building
# the image, so the container starts with a ready model.
RUN python -m ml.fetch_data && python -m ml.train && rm -rf ml/data

ENV DATABASE_URL=sqlite:////data/screenings.db
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
