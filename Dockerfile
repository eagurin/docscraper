FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
	wget \
	gnupg \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Install poetry and dependencies
RUN pip install poetry && \
	poetry config virtualenvs.create false && \
	poetry install --no-interaction --no-ansi

CMD ["poetry", "run", "python", "main.py"]
