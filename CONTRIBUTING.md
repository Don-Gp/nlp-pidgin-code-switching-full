# Contributing to Nigerian Pidgin English Code-Switching Detection

Thank you for considering contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

1. Clone the repository:
git clone https://github.com/yourusername/nlp-pidgin-code-switching.git
cd nlp-pidgin-code-switching
Copy
2. Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -e ".[dev]"
Copy
3. Install Tawa toolkit (required for PPM models)

## Running Tests

Run tests using pytest:
pytest
Copy
## Code Style

This project follows the PEP 8 style guide. Please ensure your code conforms to this style.

You can check your code style with:
flake8 src tests
Copy
And automatically format your code with:
black src tests
Copy
## Creating Notebooks

When adding Jupyter notebooks, please follow these guidelines:
- Place notebooks in the `notebooks/` directory
- Use clear, descriptive names
- Include markdown cells to explain your analysis
- Clean cell outputs before committing

## Submitting Pull Requests

1. Update the README.md if necessary
2. If you're adding a new feature, please include tests
3. Make sure all tests pass
4. Submit your pull request with a clear description of the changes

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.