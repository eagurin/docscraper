# DocScraper

An asynchronous documentation scraper and parser that transforms web documentation into structured markdown files, optimized for RAG (Retrieval-Augmented Generation) systems and AI training datasets.

[Русский](README_ru.md) | [中文](README_zh.md)

## Features

- Asynchronous web crawling with concurrent processing
- Intelligent HTML to Markdown conversion
- RAG-optimized documentation structure
- AI-enhanced content organization
- Comprehensive logging system
- Docker support with resource management

## Key Benefits

- **Clean Markdown Output**: Generates well-structured markdown files
- **RAG-Ready Format**: Optimized for training AI models
- **Hierarchy Preservation**: Maintains original documentation structure
- **Rich Metadata**: Includes context and relationships
- **Cross-References**: Preserves internal links and references

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/eagurin/docscraper.git
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
- `LOG_LEVEL`: Logging detail level (default: INFO)

### Output Structure

```plaintext
docs_output/
├── sites/              # Documentation site content
│   └── {domain}/      # Per-site documentation
└── combined/          # Unified knowledge base
	├── index.md       # Global index
	└── {domain}.md    # Domain summaries
```

## RAG Integration

Key features for RAG systems:

1. **Document Structure**
   - Consistent markdown formatting
   - Documentation hierarchy preservation
   - Rich metadata inclusion

2. **Content Processing**
   - Semantic chunking
   - Context preservation
   - Cross-reference maintenance

## Docker Support

- Resource optimization for processing
- Volume mounts for docs and logs
- Health checks enabled
- Memory limit: 2GB

## License

MIT License - see [LICENSE](LICENSE) for details