"""
Integration tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


class TestAPIEndpoints:
    """Integration tests for API"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data

    @pytest.mark.slow
    def test_sentiment_endpoint_persian(self):
        """Test sentiment analysis for Persian text"""
        payload = {
            "text": "این فیلم عالی بود",
            "language": "fa"
        }
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "sentiment_score" in data
        assert 0 <= data["sentiment_score"] <= 1

    @pytest.mark.slow
    def test_sentiment_endpoint_english(self):
        """Test sentiment analysis for English text"""
        payload = {
            "text": "This movie was great",
            "language": "en"
        }
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data

    def test_sentiment_endpoint_empty_text(self):
        """Test error on empty text"""
        payload = {
            "text": "",
            "language": "fa"
        }
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 422  # Validation error

    @pytest.mark.slow
    def test_dialect_endpoint(self):
        """Test dialect detection"""
        payload = {
            "text": "سلام داداش چطوری",
            "language": "fa"
        }
        response = client.post("/dialect", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "dialect" in data
        assert "dialect_score" in data

    def test_dialect_endpoint_wrong_language(self):
        """Test error when language is not Persian"""
        payload = {
            "text": "hello world",
            "language": "en"
        }
        response = client.post("/dialect", json=payload)
        assert response.status_code == 400

    @pytest.mark.slow
    def test_analyze_endpoint(self):
        """Test combined analysis"""
        payload = {
            "text": "این فیلم عالی بود",
            "language": "fa"
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "dialect" in data

    @pytest.mark.slow
    def test_batch_endpoint(self):
        """Test batch analysis"""
        payload = {
            "texts": [
                "این فیلم عالی بود",
                "این فیلم بد بود"
            ],
            "language": "fa"
        }
        response = client.post("/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert "total_texts" in data
        assert data["total_texts"] == 2

    def test_batch_endpoint_too_many(self):
        """Test error when batch is too large"""
        payload = {
            "texts": ["test"] * 101,
            "language": "fa"
        }
        response = client.post("/batch", json=payload)
        assert response.status_code in [400, 422]
