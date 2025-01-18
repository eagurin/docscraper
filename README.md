# DocsiteToMD

> 🌍 Language: [English](README.md) | [Русский](docs/README_ru.md) | [中文](docs/README_zh.md)

A specialized tool for converting documentation websites into structured markdown files, optimized for training RAG (Retrieval-Augmented Generation) systems. It crawls documentation sites, preserves their structure, and generates AI-enhanced markdown content.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Core Functionality

DocsiteToMD transforms documentation websites through several key stages:

1. **Documentation Site Crawling**: 
   - Asynchronous multi-threaded crawling
   - Smart rate limiting and robots.txt compliance
   - Documentation-specific content extraction

2. **Content Processing**:
   - HTML to Markdown conversion
   - Documentation hierarchy preservation
   - Metadata extraction and enrichment

3. **RAG Optimization**:
   - AI-powered content structuring
   - Semantic analysis for better retrieval
   - Context-aware organization

## 🏗 Architecture

```plaintext
DocsiteToMD/
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
   - Documentation site URL validation
   - Domain categorization
   - Rate limit configuration

2. **Processing Phase**
   - Concurrent page fetching
   - Documentation extraction
   - AI-assisted enhancement

3. **Output Phase**
   - RAG-optimized markdown generation
   - Cross-reference preservation
   - Index compilation

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/eagurin/docsitetomd.git

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
├── sites/              # Documentation site content
│   └── {domain}/      # Per-site documentation
└── combined/          # Unified knowledge base
    ├── index.md       # Global index
    └── {domain}.md    # Domain summaries
```

## 🔍 RAG Integration

Key features for RAG systems:

1. **Document Structure**
   - Consistent markdown formatting
   - Documentation hierarchy preservation
   - Rich metadata inclusion

2. **Content Processing**
   - Semantic chunking
   - Context preservation
   - Cross-reference maintenance

3. **Knowledge Organization**
   - Topic relationships
   - Documentation dependencies
   - Semantic connections

## 📚 Resources

- [Environment Setup](.env.example)
- [Docker Configuration](docker-compose.yml)
- [License](LICENSE)

## 🤝 Contributing

Contributions welcome! See our [Contributing Guide](docs/CONTRIBUTING.md) for details.
