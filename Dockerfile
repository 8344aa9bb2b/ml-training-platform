FROM python:3.12-slim

# Avoid prompt and buffer issues
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

# Install minimal system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first for minimal size and fast layer caching
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining platform dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy platform source tree
COPY . .

# Grant execution permissions to entrypoint
RUN chmod +x start_container.sh

# Expose ports:
# 7860: Hugging Face Spaces default web port
# 5000: Standard Flask dashboard port
# 8000: ZeroMQ Controller port
# 9090: Prometheus Metrics Exporter port
EXPOSE 7860 5000 8000 9090

# Launch container services
CMD ["/bin/bash", "start_container.sh"]
