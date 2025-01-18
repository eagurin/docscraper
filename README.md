# DocScraper

> 🌍 Language: [English](README.md) | [Русский](docs/README_ru.md) | [中文](docs/README_zh.md)

An intelligent documentation processing system that leverages asynchronous operations and AI to transform web-based documentation into structured, searchable knowledge bases.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Core Functionality

DocScraper systematically processes documentation through several key stages:

1. **Web Crawling**: 
   - Asynchronous multi-threaded crawling
   - Smart rate limiting and respect for robots.txt
   - Domain-specific content extraction

2. **Content Processing**:
   - Markdown conversion and standardization
   - Hierarchical document structure maintenance
   - Metadata extraction and enrichment

3. **AI Integration**:
   - OpenAI-powered content summarization
   - Semantic structure analysis
   - Context-aware document organization

## 🏗 Architecture

```plaintext
DocScraper/
├── Crawling Engine
│   ├── Async Fetcher
│   └── Content Extractor
├── Processing Pipeline
│   ├── Markdown Converter
│   ├── Structure Analyzer
│   └── AI Enhancer
└── Output Generator
	├── Site-Specific Docs
	└── Combined Knowledge Base
```

## 💾 Data Processing Flow

1. **Input Phase**
   - URL collection and validation
   - Domain categorization
   - Rate limit configuration

2. **Processing Phase**
   - Concurrent document fetching
   - Content extraction and cleaning
   - AI-assisted enhancement

3. **Output Phase**
   - Structured markdown generation
   - Cross-reference creation
   - Index compilation

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/eagurin/docscraper.git

# Setup environment
cp .env.example .env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run with Docker
docker-compose up --build

# Or run locally
python main.py
```

## ⚙️ Configuration

### Environment Variables
| Variable | Purpose | Default |
|----------|----------|----------|
| MODEL_NAME | OpenAI model selection | gpt-4 |
| OPENAI_API_KEY | API authentication | Required |
| LOG_LEVEL | Logging detail level | INFO |

### Output Structure
```plaintext
docs_output/
├── sites/              # Domain-specific content
│   └── {domain}/      # Per-site documentation
└── combined/          # Unified knowledge base
	├── index.md       # Global index
	└── {domain}.md    # Domain summaries
```

## 🔍 For AI/ML Integration

Key aspects for RAG systems:

1. **Document Structure**
   - Consistent markdown formatting
   - Clear hierarchical organization
   - Metadata-rich content

2. **Content Processing**
   - Semantic chunking
   - Context preservation
   - Cross-reference maintenance

3. **Knowledge Graph**
   - Topic relationships
   - Document dependencies
   - Semantic connections

## 📚 Resources

- [Environment Setup](.env.example)
- [Docker Configuration](docker-compose.yml)
- [License](LICENSE)

## 🤝 Contributing

Contributions welcome! See our [Contributing Guide](docs/CONTRIBUTING.md) for details.
