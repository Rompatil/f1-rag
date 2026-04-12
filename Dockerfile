FROM python:3.12-slim

WORKDIR /app

# Install lightweight deps only (no torch/sentence-transformers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# The pre-built FAISS index should be in data/vector_store/
# Build it locally first: python run.py --ingest

EXPOSE 8000

CMD ["python", "run.py", "--serve", "--host", "0.0.0.0"]
