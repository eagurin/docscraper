.PHONY: install run test lint clean help

PYTHON = poetry run python
URL ?= https://docs.browser-use.com/introduction
OUTPUT_DIR ?= docs_output
MAX_CONCURRENT ?= 3
WAIT_TIME ?= 3.0
MODEL ?= gpt-4

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies with Poetry"
	@echo "  make run           - Run the scraper with parameters:"
	@echo "    URL=<url>          Starting URL to scrape (required)"
	@echo "    OUTPUT_DIR=<dir>   Output directory (default: docs_output)"
	@echo "    MAX_CONCURRENT=<n> Maximum concurrent requests (default: 3)"
	@echo "    WAIT_TIME=<sec>    Wait time for page load (default: 3.0)"
	@echo "    MODEL=<name>       OpenAI model name (default: gpt-4)"
	@echo "  make docker-run    - Run with Docker (same parameters as above)"
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
		--wait-time $(WAIT_TIME) \
		$(if $(MODEL),--model $(MODEL))

docker-run:
	docker-compose run --rm docscraper python main.py \
		--url $(URL) \
		--output-dir $(OUTPUT_DIR) \
		--max-concurrent $(MAX_CONCURRENT) \
		--wait-time $(WAIT_TIME) \
		$(if $(MODEL),--model $(MODEL))

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
