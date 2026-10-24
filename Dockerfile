# Use Python 3.12 slim
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# FFmpeg provides the afftdn audio noise-reduction filter.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && ffmpeg -version >/dev/null \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port (if FastAPI or similar)
EXPOSE 8070

# Run app
# CMD ["python", "run.py"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8070"]
