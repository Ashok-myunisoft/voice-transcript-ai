# Use Python 3.12 slim
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port (if FastAPI or similar)
EXPOSE 8000

# Run app
# CMD ["python", "run.py"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]