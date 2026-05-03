FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py app.py bns_model.py start.sh ./
COPY .models/ .models/

# Make the startup script executable
RUN chmod +x start.sh

# Expose ports for both Streamlit and FastAPI
EXPOSE 8501 8000

CMD ["./start.sh"]
