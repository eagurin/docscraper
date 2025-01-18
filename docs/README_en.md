# DocScraper

An asynchronous documentation scraper and parser that crawls websites, extracts content, and generates structured markdown documentation.

[Русский](README_ru.md) | [中文](README_zh.md)

## Features

- Asynchronous web crawling with concurrent processing
- Domain-specific documentation organization
- Markdown generation with OpenAI assistance
- Structured output with combined documentation
- Comprehensive logging system with multiple levels
- Docker support with resource management

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/docscraper.git
cd docscraper
```

2. Set up environment:
```bash
cp .env.example .env  # Edit with your settings
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run with Docker:
```bash
docker-compose up --build
```

Or run locally:
```bash
python main.py
```

## Configuration

### Environment Variables (.env)
- `MODEL_NAME`: OpenAI model to use (default: gpt-4)
- `OPENAI_API_KEY`: Your OpenAI API key

### Logging
- Console (INFO level): Colored, real-time updates
- File (DEBUG level): Detailed diagnostics in `logs/docparser_{time}.log`

## Output Structure

```
docs_output/
├── sites/              # Domain-specific documentation
│   └── {domain}/      # Per-domain content
└── combined/          # Aggregated documentation
	├── index.md       # Master index
	└── {domain}.md    # Domain compilations
```

## Docker Support

- Resource limits: 2GB memory
- Volume mounts for docs and logs
- Health checks enabled

## License

MIT License - see [LICENSE](LICENSE) for details