.PHONY: install run test lint clean help

PYTHON = poetry run python
URL ?= https://docs.browser-use.com/introduction
OUTPUT_DIR ?= docs_output
MAX_CONCURRENT ?= 3
MODEL_NAME ?= gpt-4

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies with Poetry"
	@echo "  make run           - Run the scraper with parameters:"
	@echo "    URL=<url>          Starting URL to scrape (required)"
	@echo "    OUTPUT_DIR=<dir>   Output directory (default: docs_output)"
	@echo "    MAX_CONCURRENT=<n> Maximum concurrent requests (default: 3)"
	@echo "    MODEL_NAME=<name>  OpenAI model name (default: gpt-4)"
	@echo "  make lint          - Run all linters"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean up generated files"

install:
	poetry install

run:
	$(PYTHON) main.py \
		--url $(URL) \
		--output-dir $(OUTPUT_DIR) \
		--max-concurrent $(MAX_CONCURRENT) \
		$(if $(MODEL_NAME),--model $(MODEL_NAME))


lint:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .
	$(PYTHON) -m mypy .
	$(PYTHON) -m ruff .

test:
	$(PYTHON) -m pytest

clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf logs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# poetry run python main.py --url "https://www.mongodb.com/docs/drivers/python-drivers/" --output-dir /Users/laptop/dev/docscraper/output --max-concurrent 25