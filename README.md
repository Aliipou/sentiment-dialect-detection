# Multilingual Sentiment Analysis & Persian Dialect Detection

[![CI/CD](https://github.com/aliipou/sentiment-dialect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/aliipou/sentiment-dialect-detection/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


A production-ready NLP system for multilingual sentiment analysis and Persian dialect detection, built with transformers, FastAPI, and Streamlit.

## Features

- **Multilingual Sentiment Analysis**: Supports 19+ languages including Persian, English, French, German, Spanish, and more
- **Persian Dialect Detection**: Identifies common Persian dialects (Tehrani, Isfahani, Shirazi, Mashhadi, and others)
- **RESTful API**: FastAPI-based API with automatic documentation and validation
- **Web Interface**: Interactive Streamlit UI for easy text analysis
- **Production Ready**: Docker support, CI/CD pipeline, comprehensive tests
- **Well-Architected**: Clean code structure following software engineering best practices

## Quick Start

### Prerequisites

- **Python 3.8-3.11** or **Python 3.13+** (with fallback mode)
  - Python 3.8-3.11: Full Persian NLP support with hazm
  - Python 3.13+: Works with fallback Persian processing (hazm compatibility pending)
- 4GB RAM minimum (8GB+ recommended)
- Optional: CUDA-compatible GPU for faster inference

> **Note**: The project works on all Python versions. Python 3.13 uses fallback Persian text processing until hazm library updates for compatibility.

### Installation

#### Option 1: Using Make (Recommended)

```bash
# Clone the repository
git clone https://github.com/aliipou/sentiment-dialect-detection.git
cd sentiment-dialect-detection

# Setup development environment
make setup

# Install dependencies
make install-dev
```

#### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/aliipou/sentiment-dialect-detection.git
cd sentiment-dialect-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
```

#### Option 3: Using Docker

```bash
# Clone the repository
git clone https://github.com/aliipou/sentiment-dialect-detection.git
cd sentiment-dialect-detection

# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

## Usage

### Running the API

```bash
# Using make
make run-api

# Or directly with Python
python scripts/run_api.py

# Or with uvicorn
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

API will be available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### API Examples

#### Sentiment Analysis

```bash
curl -X POST "http://localhost:8000/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was excellent!", "language": "en"}'
```

#### Dialect Detection

```bash
curl -X POST "http://localhost:8000/dialect" \
  -H "Content-Type: application/json" \
  -d '{"text": "سلام داداش، چطوری؟", "language": "fa"}'
```

#### Combined Analysis

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "این فیلم عالی بود", "language": "fa"}'
```

#### Batch Processing

```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["این فیلم عالی بود", "این فیلم بد بود"], "language": "fa"}'
```

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run with coverage report
pytest --cov=src --cov-report=html
```

## Project Structure

```
sentiment-dialect-detection/
├── src/
│   ├── api/              # FastAPI application
│   │   ├── app.py        # Main API app
│   │   ├── schemas.py    # Pydantic schemas
│   │   └── service.py    # Business logic
│   ├── models/           # ML models
│   │   ├── base.py       # Base model classes
│   │   ├── sentiment.py  # Persian sentiment model
│   │   ├── multilingual.py  # Multilingual sentiment model
│   │   └── dialect.py    # Dialect detection model
│   ├── preprocessing/    # Data preprocessing
│   │   ├── text_processor.py  # Text cleaning
│   │   └── data_loader.py     # Data loading
│   ├── utils/            # Utilities
│   │   ├── config.py     # Configuration management
│   │   └── logger.py     # Logging setup
│   └── web/              # Streamlit web interface
├── tests/
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── data/                 # Training data
├── models/               # Trained models
├── logs/                 # Application logs
├── scripts/              # Utility scripts
├── .github/workflows/    # CI/CD pipelines
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose config
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── setup.py              # Package setup
├── pyproject.toml        # Project configuration
├── Makefile              # Common tasks
└── README.md             # This file
```

## Development

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Models

- **Persian Sentiment**: ParsBERT (HooshvareLab/bert-fa-base-uncased)
- **Multilingual Sentiment**: XLM-RoBERTa (xlm-roberta-base)
- **Dialect Detection**: TF-IDF + Naive Bayes with rule-based fallback

## Configuration

Configuration is managed through environment variables and `.env` file:

```bash
# Model Paths
SENTIMENT_MODEL_PATH=models/sentiment_model
DIALECT_MODEL_PATH=models/dialect_model.joblib
MULTILINGUAL_MODEL_PATH=models/multilingual_model

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2

# Performance
USE_GPU=true
MAX_BATCH_SIZE=100
```

See `.env.example` for all available options.

## Deployment

### Docker Production Deployment

```bash
# Build production image
docker build --target production -t sentiment-api:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  --env-file .env \
  --name sentiment-api \
  sentiment-api:latest
```

## Supported Languages

Persian (fa), English (en), French (fr), German (de), Spanish (es), Italian (it), Portuguese (pt), Dutch (nl), Swedish (sv), Norwegian (no), Danish (da), Finnish (fi), Greek (el), Russian (ru), Polish (pl), Czech (cs), Hungarian (hu), Romanian (ro), Turkish (tr)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- ParsBERT by HooshvareLab
- XLM-RoBERTa by Facebook AI
- Hazm Persian NLP library
- FastAPI and Streamlit communities

---

Made with ❤️ for NLP
