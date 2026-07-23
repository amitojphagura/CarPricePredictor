FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer across rebuilds
# that only change application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, pre-trained model artifacts, and the source data (needed
# only if you retrain inside the container -- see below).
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

# Serves predictions using the model/preprocessor already baked into the
# image (models/model.pkl, models/preprocessor.pkl). To retrain instead,
# run: docker run <image> python -m src.pipeline.train_pipeline
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.main:app"]
