# Contributing to DocScraper

We welcome contributions to DocScraper! This document provides guidelines and best practices for contributing to the project.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/docscraper.git
cd docscraper

# Set up development environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write descriptive docstrings
- Keep functions focused and modular

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Include integration tests where appropriate

## Pull Request Process

1. Update documentation for new features
2. Add tests for new functionality
3. Update CHANGELOG.md if applicable
4. Ensure CI passes all checks
5. Request review from maintainers

## Documentation

- Keep README.md updated
- Document new features
- Include docstrings for public APIs
- Update architecture diagrams if needed

## Questions?

Feel free to open an issue for:
- Feature proposals
- Bug reports
- Documentation improvements
- General questions

Thank you for contributing to DocScraper!