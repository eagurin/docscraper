# Use Python 3.8 slim image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
	chromium \
	chromium-driver \
	&& rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p docs_output/sites docs_output/combined logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BROWSER_EXECUTABLE_PATH=/usr/bin/chromium

# Run the application
CMD ["python", "main.py"]