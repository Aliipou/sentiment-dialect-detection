# Quick Start Guide

## Installation & Testing

### Step 1: Install Dependencies

```bash
# Activate virtual environment (if you have one)
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Unit Tests (No Models Required)

```bash
# Run unit tests only (fast, no model loading)
pytest tests/unit/ -v
```

### Step 3: Start the API

```bash
# Start API server
python scripts/run_api.py
```

### Step 4: Test API

Open another terminal and test:

```bash
# Health check
curl http://localhost:8000/health

# Test sentiment (will use pretrained models - first run will download them)
curl -X POST "http://localhost:8000/sentiment" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This is great!\", \"language\": \"en\"}"
```

## Pushing to GitHub

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository named `sentiment-dialect-detection`
3. Do NOT initialize with README (we already have one)

### Step 2: Push Code

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "feat: complete rebuild with professional structure

- Refactored into clean module structure (src/api, src/models, etc.)
- Added comprehensive unit and integration tests
- Added Docker support and docker-compose
- Added CI/CD pipeline with GitHub Actions
- Added pre-commit hooks for code quality
- Updated documentation with detailed README
- Added configuration management with pydantic-settings
- Implemented proper logging throughout
- Added Makefile for common tasks
"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/sentiment-dialect-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Update README

After pushing, update the README.md file:

1. Replace all instances of `yourusername` with your actual GitHub username
2. Commit and push:

```bash
git add README.md
git commit -m "docs: update GitHub username in README"
git push
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, make sure you:
1. Activated virtual environment
2. Installed dependencies: `pip install -r requirements.txt`

### Model Download Issues

First time running will download large models (~500MB each):
- ParsBERT for Persian sentiment
- XLM-RoBERTa for multilingual

Make sure you have:
- Good internet connection
- Sufficient disk space (~2GB free)

### Port Already in Use

If port 8000 is busy:
```bash
# Find process using port 8000 (Windows)
netstat -ano | findstr :8000

# Kill the process
taskkill /PID <PID> /F

# Or use a different port
uvicorn src.api.app:app --port 8001
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Run tests
3. ✅ Start API and test
4. ✅ Push to GitHub
5. ⏭️ Configure environment variables in `.env`
6. ⏭️ Train models with your own data (optional)
7. ⏭️ Deploy with Docker
