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

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Set up development environment
poetry install --with dev
poetry shell
```

## Code Style

We use several tools to maintain code quality:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- ruff for linting

Run all checks with:
```bash
poetry run black .
poetry run isort .
poetry run mypy .
poetry run ruff .
```

## Testing

- Write unit tests for new features
- Use pytest and pytest-asyncio for testing
- Run tests with: `poetry run pytest`
- Include integration tests where appropriate

## Pull Request Process

1. Update documentation for new features
2. Add tests for new functionality
3. Update CHANGELOG.md if applicable
4. Ensure all code quality checks pass
5. Request review from maintainers

## Documentation

- Keep README.md and translations (README_ru.md, README_zh.md) updated
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