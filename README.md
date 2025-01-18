# 🚀 DocScraper

> Transform documentation websites into RAG-optimized markdown collections for AI training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🌍 **Languages**: [English](README.md) | [Русский](docs/README_ru.md) | [中文](docs/README_zh.md)

## 🎯 What it Does

DocScraper automatically converts documentation websites into clean, structured markdown files optimized for RAG (Retrieval-Augmented Generation) systems and AI training datasets. Perfect for creating high-quality training data for your AI models.

### ✨ Key Features

- 🔄 **Smart Crawling**: Asynchronous, multi-threaded website processing
- 📝 **Intelligent Conversion**: HTML → Clean Markdown transformation
- 🧠 **AI Enhancement**: OpenAI-powered content structuring
- 📊 **RAG Optimization**: Perfect for training data preparation
- 🔍 **Metadata Rich**: Preserves context and relationships
- 🐳 **Docker Ready**: Easy deployment and scaling

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
- 🐳 Docker (optional)

### 📦 Installation

1. **Clone and Setup**:
```bash
git clone https://github.com/eagurin/docscraper.git
cd docscraper
```

2. **Create Environment**:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure**:
```bash
cp .env.example .env
# Edit .env with your settings:
# MODEL_NAME=gpt-4
# OPENAI_API_KEY=your_key_here
```

### 🎮 Usage

**With Docker** (recommended):
```bash
docker-compose up --build
```

**Without Docker**:
```bash
python main.py
```

## 📁 Project Structure

```plaintext
docscraper/
├── 📂 docs_output/        # Generated documentation
│   ├── sites/           # Per-site content
│   └── combined/        # Unified knowledge base
├── 📝 main.py           # Core application
├── 🐳 Dockerfile        # Container config
├── 📋 requirements.txt  # Dependencies
└── ⚙️ .env             # Configuration
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
- 🖥️ Memory Limit: 2GB
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