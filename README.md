# 🚀 DocScraper

> Transform documentation websites into RAG-optimized markdown collections for AI training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🌍 **Languages**: [English](README.md) | [Русский](README_ru.md) | [中文](README_zh.md)

## 🎯 What it Does

DocScraper automatically converts documentation websites into clean, structured markdown files optimized for RAG (Retrieval-Augmented Generation) systems and AI training datasets. Perfect for creating high-quality training data for your AI models.

### ✨ Key Features

- 🔄 **Smart Crawling**: Asynchronous, multi-threaded website processing
- 📝 **Intelligent Conversion**: HTML → Clean Markdown transformation
- 🧠 **AI Enhancement**: OpenAI-powered content structuring
- 📊 **RAG Optimization**: Perfect for training data preparation
- 🔍 **Metadata Rich**: Preserves context and relationships
- 🐳 **Docker-ready**: Easy deployment and scaling


## 💫 Why DocScraper?

- 📚 **Clean Documentation**: Perfectly formatted markdown files
- 🤖 **AI-Ready Format**: Optimized for RAG systems
- 🌳 **Structure Preservation**: Maintains original hierarchy
- 🔗 **Smart References**: Keeps internal links and context
- 🎨 **Rich Metadata**: Enhanced with AI-generated insights

## 🚀 Quick Start

### Prerequisites

- 🐍 Python 3.8+
- 🔑 OpenAI API key

### 📦 Installation

1. **Install with Poetry**:
```bash
poetry install
```

2. **Configure**:
```bash
cp .env.example .env
# Edit .env with your settings:
# MODEL_NAME=gpt-4
# OPENAI_API_KEY=your_key_here
```

### 🎮 Usage

```bash
# Basic usage
make run URL=https://your-docs.com

# Advanced usage with all parameters
make run \
	URL=https://your-docs.com \
	OUTPUT_DIR=custom_docs \
	MAX_CONCURRENT=5 \
	WAIT_TIME=5.0 \
	MODEL=gpt-4
```

### ⚙️ Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| URL | Starting URL to scrape | Required |
| OUTPUT_DIR | Output directory | docs_output |
| MAX_CONCURRENT | Maximum concurrent requests | 3 |
| WAIT_TIME | Wait time for page load (seconds) | 3.0 |
| MODEL | OpenAI model name | gpt-4 |

## 🛠 Development

### Setup Development Environment
```bash
poetry install --with dev
poetry shell
```

### Code Quality
```bash
poetry run black .
poetry run isort .
poetry run mypy .
poetry run ruff .
```

### Testing
```bash
poetry run pytest
```

## 📁 Project Structure

```plaintext
docscraper/
├── 📂 docs_output/        # Generated documentation
│   ├── sites/           # Per-site content
│   └── combined/        # Unified knowledge base
├── 📝 main.py           # Core application
├── 📄 pyproject.toml    # Project configuration
└── ⚙️ .env             # Environment configuration
```

## 🎨 Output Format

DocScraper generates two types of RAG-optimized content:

### 1. 📑 Site-Specific Documentation
- Clean markdown per page
- Original URL structure
- Rich metadata headers
- AI-enhanced content

### 2. 📚 Combined Knowledge Base
- Cross-referenced documentation
- Global search index
- Topic relationships
- Semantic connections

## ⚙️ Configuration

### Environment Variables
| Variable | Purpose | Default |
|----------|---------|---------|
| MODEL_NAME | OpenAI model | gpt-4 |
| OPENAI_API_KEY | API authentication | Required |
| LOG_LEVEL | Logging detail | INFO |
| MAX_CONCURRENT | Parallel operations | 3 |

### 🔧 Resource Settings
- 📊 Concurrent Tasks: 3
- 📝 Log Rotation: 500MB
- 🕒 Log Retention: 10 days

## 📈 RAG Integration

### Document Processing
- 📝 Consistent markdown formatting
- 🌳 Hierarchical structure
- 🏷️ Rich metadata inclusion
- 🔍 Semantic chunking
- 🔗 Cross-references

### Knowledge Organization
- 📚 Topic relationships
- 🔄 Document dependencies
- 🧩 Semantic connections

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](docs/CONTRIBUTING.md) for:
- 📝 Code style guidelines
- 🔍 Testing requirements
- 🚀 PR process
- 📦 Development setup

## 📄 License

MIT License - See [LICENSE](LICENSE) for details
