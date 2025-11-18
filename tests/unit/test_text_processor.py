"""
Unit tests for text preprocessing
"""
import pytest
from src.preprocessing.text_processor import (
    PersianTextPreprocessor,
    EnglishTextPreprocessor,
    MultilingualTextPreprocessor,
    get_preprocessor
)


class TestPersianTextPreprocessor:
    """Tests for Persian text preprocessing"""

    def test_clean_text_removes_non_persian(self):
        processor = PersianTextPreprocessor()
        text = "سلام123!@#world"
        cleaned = processor.clean_text(text)
        assert "123" not in cleaned
        assert "world" not in cleaned
        assert "سلام" in cleaned

    def test_clean_text_normalizes(self):
        processor = PersianTextPreprocessor()
        text = "ﺳﻼﻡ"  # Non-standard Persian characters
        cleaned = processor.clean_text(text)
        assert len(cleaned) > 0

    def test_clean_text_removes_extra_spaces(self):
        processor = PersianTextPreprocessor()
        text = "سلام    دنیا"
        cleaned = processor.clean_text(text)
        assert "    " not in cleaned
        assert "سلام دنیا" == cleaned

    def test_clean_text_empty_input(self):
        processor = PersianTextPreprocessor()
        assert processor.clean_text("") == ""
        assert processor.clean_text(None) == ""
        assert processor.clean_text(123) == ""

    def test_tokenize(self):
        processor = PersianTextPreprocessor()
        text = "سلام دنیا"
        tokens = processor.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) >= 2


class TestEnglishTextPreprocessor:
    """Tests for English text preprocessing"""

    def test_clean_text_lowercase(self):
        processor = EnglishTextPreprocessor()
        text = "Hello WORLD"
        cleaned = processor.clean_text(text)
        assert cleaned == "hello world"

    def test_clean_text_removes_numbers(self):
        processor = EnglishTextPreprocessor()
        text = "hello123world"
        cleaned = processor.clean_text(text)
        assert "123" not in cleaned

    def test_clean_text_removes_special_chars(self):
        processor = EnglishTextPreprocessor()
        text = "hello!@#$world"
        cleaned = processor.clean_text(text)
        assert "!@#$" not in cleaned
        assert "hello" in cleaned
        assert "world" in cleaned

    def test_tokenize(self):
        processor = EnglishTextPreprocessor()
        text = "hello world"
        tokens = processor.tokenize(text)
        assert tokens == ["hello", "world"]


class TestMultilingualTextPreprocessor:
    """Tests for multilingual text preprocessing"""

    def test_detect_language_persian(self):
        processor = MultilingualTextPreprocessor()
        text = "سلام دنیا"
        lang = processor.detect_language(text)
        assert lang == "fa"

    def test_detect_language_english(self):
        processor = MultilingualTextPreprocessor()
        text = "hello world"
        lang = processor.detect_language(text)
        assert lang == "en"

    def test_detect_language_empty(self):
        processor = MultilingualTextPreprocessor()
        lang = processor.detect_language("")
        assert lang == "en"  # Default

    def test_clean_text_auto_detect(self):
        processor = MultilingualTextPreprocessor()
        persian_text = "سلام123"
        english_text = "hello123"

        cleaned_fa = processor.clean_text(persian_text)
        cleaned_en = processor.clean_text(english_text)

        assert "سلام" in cleaned_fa
        assert "123" not in cleaned_fa
        assert "hello" in cleaned_en
        assert "123" not in cleaned_en


class TestGetPreprocessor:
    """Tests for preprocessor factory function"""

    def test_get_persian_preprocessor(self):
        processor = get_preprocessor("fa")
        assert isinstance(processor, PersianTextPreprocessor)

    def test_get_english_preprocessor(self):
        processor = get_preprocessor("en")
        assert isinstance(processor, EnglishTextPreprocessor)

    def test_get_multilingual_preprocessor(self):
        processor = get_preprocessor("auto")
        assert isinstance(processor, MultilingualTextPreprocessor)

    def test_get_unknown_language(self):
        processor = get_preprocessor("unknown")
        assert isinstance(processor, EnglishTextPreprocessor)
