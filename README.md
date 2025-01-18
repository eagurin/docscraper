# DocScraper

An asynchronous documentation scraper and parser that transforms web documentation into structured markdown files, optimized for RAG (Retrieval-Augmented Generation) systems and AI training datasets.

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

## Requirements

- Python 3.8+
- OpenAI API key
- Docker (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/docscraper.git
cd docscraper
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file:
```bash
MODEL_NAME=gpt-4
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running with Python

```bash
python main.py
```

### Running with Docker Compose

```bash
docker-compose up --build
```

## Project Structure

```
docscraper/
├── docs_output/          # Output directory for documentation
│   ├── sites/           # Domain-specific documentation
│   └── combined/        # Combined documentation files
├── logs/                # Log files directory
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
└── .env                # Environment variables
```

## Output Structure

The parser generates RAG-optimized markdown in two formats:

1. Domain-specific documentation (`docs_output/sites/{domain}/`):
   - Clean markdown files for each page
   - Preserved URL structure in markdown
   - Metadata-rich content

2. Combined documentation (`docs_output/combined/`):
   - Domain-specific markdown compilations
   - Master markdown index
   - Cross-referenced documentation

## Logging

The application uses loguru for comprehensive logging:

- Console output (INFO level):
  - Colored formatting
  - Real-time processing updates
  - Important operational information

- File logging (DEBUG level):
  - Detailed debug information
  - Full error tracebacks
  - Diagnostic information
  - Log rotation: 500MB per file
  - Log retention: 10 days

Log files are stored in `logs/docparser_{time}.log`

## Configuration

Key configuration options:

- `MAX_CONCURRENT`: Maximum concurrent operations (default: 3)
- `BROWSER_CONFIG`: Browser configuration settings
- Docker resource limits:
  - Memory limit: 2GB
  - Memory reservation: 1GB
- Log levels:
  - Console: INFO
  - File: DEBUG

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request