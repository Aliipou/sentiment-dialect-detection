"""
Unit tests for API schemas
"""
import pytest
from pydantic import ValidationError
from src.api.schemas import TextInput, BatchTextInput, SentimentResponse


class TestTextInput:
    """Tests for TextInput schema"""

    def test_valid_input(self):
        """Test valid text input"""
        data = TextInput(text="test text", language="fa")
        assert data.text == "test text"
        assert data.language == "fa"

    def test_default_language(self):
        """Test default language"""
        data = TextInput(text="test")
        assert data.language == "fa"

    def test_empty_text_error(self):
        """Test error on empty text"""
        with pytest.raises(ValidationError):
            TextInput(text="")

    def test_whitespace_only_text_error(self):
        """Test error on whitespace-only text"""
        with pytest.raises(ValidationError):
            TextInput(text="   ")

    def test_text_strip(self):
        """Test text stripping"""
        data = TextInput(text="  test  ")
        assert data.text == "test"

    def test_invalid_language(self):
        """Test error on invalid language"""
        with pytest.raises(ValidationError):
            TextInput(text="test", language="invalid")

    def test_too_long_text_error(self):
        """Test error on text that's too long"""
        with pytest.raises(ValidationError):
            TextInput(text="x" * 10001)


class TestBatchTextInput:
    """Tests for BatchTextInput schema"""

    def test_valid_batch_input(self):
        """Test valid batch input"""
        data = BatchTextInput(texts=["text1", "text2"], language="en")
        assert len(data.texts) == 2
        assert data.language == "en"

    def test_empty_list_error(self):
        """Test error on empty list"""
        with pytest.raises(ValidationError):
            BatchTextInput(texts=[])

    def test_too_many_texts_error(self):
        """Test error on too many texts"""
        with pytest.raises(ValidationError):
            BatchTextInput(texts=["text"] * 101)

    def test_texts_strip(self):
        """Test that texts are stripped"""
        data = BatchTextInput(texts=["  text1  ", "  text2  "])
        assert data.texts == ["text1", "text2"]

    def test_all_empty_texts_error(self):
        """Test error when all texts are empty"""
        with pytest.raises(ValidationError):
            BatchTextInput(texts=["", "  ", ""])


class TestSentimentResponse:
    """Tests for SentimentResponse schema"""

    def test_valid_response(self):
        """Test valid sentiment response"""
        data = SentimentResponse(
            text="test",
            sentiment="positive",
            sentiment_score=0.95,
            processing_time=0.1
        )
        assert data.sentiment == "positive"
        assert data.sentiment_score == 0.95

    def test_invalid_score_range(self):
        """Test error on invalid score range"""
        with pytest.raises(ValidationError):
            SentimentResponse(
                text="test",
                sentiment="positive",
                sentiment_score=1.5,
                processing_time=0.1
            )
