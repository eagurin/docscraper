# Use Python 3.8 slim image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y \
	chromium \
	chromium-driver \
	curl \
	&& rm -rf /var/lib/apt/lists/* \
	&& curl -sSL https://install.python-poetry.org | python3 -

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
	&& poetry install --no-interaction --no-ansi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p docs_output/sites docs_output/combined logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BROWSER_EXECUTABLE_PATH=/usr/bin/chromium

# Run the application
CMD ["poetry", "run", "python", "main.py"]