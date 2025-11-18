"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_persian_texts():
    """Sample Persian texts for testing"""
    return [
        "این فیلم واقعاً عالی بود، از دیدنش لذت بردم.",
        "کیفیت محصول متوسط بود، نه خوب نه بد.",
        "خدمات بسیار ضعیف بود و قیمت‌ها خیلی گران."
    ]


@pytest.fixture
def sample_english_texts():
    """Sample English texts for testing"""
    return [
        "This movie was amazing, I really enjoyed it.",
        "The product quality was average, neither good nor bad.",
        "The service was terrible and the prices were too high."
    ]


@pytest.fixture
def sample_dialect_texts():
    """Sample Persian dialect texts"""
    return {
        'تهرانی': "سلام داداش، چطوری؟ دیشب کجا بودی؟",
        'اصفهانی': "زِدی به بیابون و همه چی رو خراب کردی؟",
        'سایر': "لطفاً این موضوع را بررسی کنید."
    }


@pytest.fixture
def sentiment_labels():
    """Sentiment label mappings"""
    return {
        'fa': {0: "منفی", 1: "خنثی", 2: "مثبت"},
        'en': {0: "negative", 1: "neutral", 2: "positive"}
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary models directory"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir
