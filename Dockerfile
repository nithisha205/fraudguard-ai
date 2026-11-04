FROM python:3.11-slim

# XGBoost needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY .streamlit/ .streamlit/
COPY src/ src/
COPY models/fraud_model.joblib models/
COPY samples/ samples/
COPY reports/monitoring.json reports/
COPY reports/figures/threshold_cost.png reports/figures/monitoring.png reports/figures/

EXPOSE 8000 8501

# Serve the API by default; docker-compose.yml also runs the dashboard from this image.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
