FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directory
RUN mkdir -p data

EXPOSE 8000

# Railway manages its own healthcheck via railway.toml — Docker HEALTHCHECK disabled
# to avoid conflict.

CMD ["python", "main.py"]
