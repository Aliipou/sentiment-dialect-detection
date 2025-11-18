"""
Text preprocessing utilities for multiple languages
"""
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

# Try to import hazm, but make it optional
try:
    from hazm import Normalizer, word_tokenize
    HAZM_AVAILABLE = True
    logger.info("hazm library loaded successfully")
except ImportError:
    HAZM_AVAILABLE = False
    logger.warning("hazm library not available - using fallback Persian processor")
    Normalizer = None
    word_tokenize = None


class TextPreprocessor(ABC):
    """Abstract base class for text preprocessing"""

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        pass


class PersianTextPreprocessor(TextPreprocessor):
    """Persian text preprocessing using Hazm (with fallback)"""

    def __init__(self):
        if HAZM_AVAILABLE:
            self.normalizer = Normalizer()
            logger.info("Initialized Persian text preprocessor with hazm")
        else:
            self.normalizer = None
            logger.info("Initialized Persian text preprocessor with fallback (no hazm)")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Persian text

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        try:
            # Normalize Persian text if hazm available
            if HAZM_AVAILABLE and self.normalizer:
                text = self.normalizer.normalize(text)
            else:
                # Fallback normalization
                text = self._fallback_normalize(text)

            # Remove non-Persian characters (keep Persian Unicode range + spaces)
            text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        except Exception as e:
            logger.error(f"Error cleaning Persian text: {e}")
            return ""

    def _fallback_normalize(self, text: str) -> str:
        """Fallback normalization without hazm"""
        # Basic Arabic/Persian character normalization
        replacements = {
            'ك': 'ک', 'ي': 'ی', 'ى': 'ی',
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Persian text

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        try:
            if HAZM_AVAILABLE and word_tokenize:
                return word_tokenize(text)
            else:
                # Fallback: simple split
                return text.split()
        except Exception as e:
            logger.error(f"Error tokenizing Persian text: {e}")
            return text.split()


class EnglishTextPreprocessor(TextPreprocessor):
    """English text preprocessing"""

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize English text

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        try:
            # Convert to lowercase
            text = text.lower()

            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        except Exception as e:
            logger.error(f"Error cleaning English text: {e}")
            return ""

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize English text

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return text.split()


class MultilingualTextPreprocessor:
    """Preprocessor that handles multiple languages"""

    def __init__(self):
        self.persian_preprocessor = PersianTextPreprocessor()
        self.english_preprocessor = EnglishTextPreprocessor()
        logger.info("Initialized multilingual text preprocessor")

    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character sets

        Args:
            text: Input text

        Returns:
            Language code ('fa' or 'en')
        """
        if not text:
            return "en"

        # Check for Persian characters
        persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        text_chars = set(text.lower())

        if any(char in persian_chars for char in text_chars):
            return "fa"

        return "en"

    def clean_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Clean text with automatic language detection

        Args:
            text: Input text
            language: Language code (auto-detected if None)

        Returns:
            Cleaned text
        """
        if language is None:
            language = self.detect_language(text)

        if language == "fa":
            return self.persian_preprocessor.clean_text(text)
        else:
            return self.english_preprocessor.clean_text(text)

    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize text with automatic language detection

        Args:
            text: Input text
            language: Language code (auto-detected if None)

        Returns:
            List of tokens
        """
        if language is None:
            language = self.detect_language(text)

        if language == "fa":
            return self.persian_preprocessor.tokenize(text)
        else:
            return self.english_preprocessor.tokenize(text)


def get_preprocessor(language: str = "auto") -> TextPreprocessor:
    """
    Factory function to get appropriate preprocessor

    Args:
        language: Language code or 'auto'

    Returns:
        Text preprocessor instance
    """
    if language == "fa":
        return PersianTextPreprocessor()
    elif language == "en":
        return EnglishTextPreprocessor()
    elif language == "auto":
        return MultilingualTextPreprocessor()
    else:
        logger.warning(f"Unknown language '{language}', using English preprocessor")
        return EnglishTextPreprocessor()
