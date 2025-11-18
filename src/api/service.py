"""
Business logic for API endpoints
"""
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..models.sentiment import PersianSentimentAnalyzer, SENTIMENT_LABELS_FA
from ..models.multilingual import MultilingualSentimentAnalyzer, SENTIMENT_LABELS
from ..models.dialect import PersianDialectDetector
from ..utils.config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class AnalysisService:
    """Service class for sentiment and dialect analysis"""

    def __init__(self):
        self.sentiment_analyzer = None
        self.multilingual_analyzer = None
        self.dialect_detector = None
        self._models_loaded = {
            'sentiment': False,
            'multilingual': False,
            'dialect': False
        }

    async def initialize(self):
        """Initialize and load all models"""
        logger.info("Initializing analysis service...")

        try:
            # Load Persian sentiment analyzer
            self.sentiment_analyzer = PersianSentimentAnalyzer()
            if Path(settings.SENTIMENT_MODEL_PATH).exists():
                try:
                    self.sentiment_analyzer.load(settings.SENTIMENT_MODEL_PATH)
                    self._models_loaded['sentiment'] = True
                    logger.info("Persian sentiment model loaded from trained checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load trained sentiment model: {e}")
                    self.sentiment_analyzer.load_tokenizer()
                    self.sentiment_analyzer.load_model()
                    logger.info("Using pretrained Persian sentiment model")
            else:
                self.sentiment_analyzer.load_tokenizer()
                self.sentiment_analyzer.load_model()
                logger.info("Using pretrained Persian sentiment model")

            # Load multilingual analyzer
            self.multilingual_analyzer = MultilingualSentimentAnalyzer()
            if Path(settings.MULTILINGUAL_MODEL_PATH).exists():
                try:
                    self.multilingual_analyzer.load(settings.MULTILINGUAL_MODEL_PATH)
                    self._models_loaded['multilingual'] = True
                    logger.info("Multilingual model loaded from trained checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load trained multilingual model: {e}")
                    self.multilingual_analyzer.load_tokenizer()
                    self.multilingual_analyzer.load_model()
                    logger.info("Using pretrained multilingual model")
            else:
                self.multilingual_analyzer.load_tokenizer()
                self.multilingual_analyzer.load_model()
                logger.info("Using pretrained multilingual model")

            # Load dialect detector
            self.dialect_detector = PersianDialectDetector()
            if Path(settings.DIALECT_MODEL_PATH).exists():
                try:
                    self.dialect_detector.load(settings.DIALECT_MODEL_PATH)
                    self._models_loaded['dialect'] = True
                    logger.info("Dialect model loaded from trained checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load trained dialect model: {e}")
                    logger.info("Using rule-based dialect detection")
            else:
                logger.info("Using rule-based dialect detection")

            logger.info("Analysis service initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing analysis service: {e}")
            raise

    def get_models_status(self) -> Dict[str, bool]:
        """Get status of loaded models"""
        return self._models_loaded.copy()

    async def analyze_sentiment(
        self,
        text: str,
        language: str = "fa"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text

        Args:
            text: Input text
            language: Language code

        Returns:
            Sentiment analysis result
        """
        start_time = time.time()

        try:
            # Select appropriate analyzer
            if language == "fa":
                analyzer = self.sentiment_analyzer
                lang_labels = SENTIMENT_LABELS_FA
            else:
                analyzer = self.multilingual_analyzer
                lang_labels = SENTIMENT_LABELS.get(language, SENTIMENT_LABELS["en"])

            # Predict
            prediction = analyzer.predict(text, return_probabilities=True)

            # Extract results
            label = prediction['labels'][0]
            sentiment_name = lang_labels[label]
            confidence_score = float(prediction['probabilities'][0][label])

            processing_time = time.time() - start_time

            return {
                'text': text,
                'sentiment': sentiment_name,
                'sentiment_score': confidence_score,
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise

    async def analyze_dialect(self, text: str) -> Dict[str, Any]:
        """
        Analyze dialect of Persian text

        Args:
            text: Input text

        Returns:
            Dialect analysis result
        """
        start_time = time.time()

        try:
            prediction = self.dialect_detector.predict(text)

            dialect_label = prediction['labels'][0]
            dialect_name = prediction['dialect_names'][0]

            # Get confidence score
            if prediction['probabilities'] and prediction['probabilities'][0]:
                confidence_score = float(prediction['probabilities'][0][0])
            else:
                confidence_score = 0.5

            processing_time = time.time() - start_time

            return {
                'text': text,
                'dialect': dialect_name,
                'dialect_score': confidence_score,
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Error in dialect analysis: {e}")
            raise

    async def analyze_combined(
        self,
        text: str,
        language: str = "fa"
    ) -> Dict[str, Any]:
        """
        Combined sentiment and dialect analysis

        Args:
            text: Input text
            language: Language code

        Returns:
            Combined analysis result
        """
        start_time = time.time()

        try:
            # Sentiment analysis
            if language == "fa":
                analyzer = self.sentiment_analyzer
                lang_labels = SENTIMENT_LABELS_FA
            else:
                analyzer = self.multilingual_analyzer
                lang_labels = SENTIMENT_LABELS.get(language, SENTIMENT_LABELS["en"])

            sentiment_pred = analyzer.predict(text, return_probabilities=True)
            sentiment_label = sentiment_pred['labels'][0]
            sentiment_name = lang_labels[sentiment_label]
            sentiment_score = float(sentiment_pred['probabilities'][0][sentiment_label])

            # Dialect analysis (only for Persian)
            dialect_name = None
            dialect_score = None

            if language == "fa":
                dialect_pred = self.dialect_detector.predict(text)
                dialect_name = dialect_pred['dialect_names'][0]
                if dialect_pred['probabilities'] and dialect_pred['probabilities'][0]:
                    dialect_score = float(dialect_pred['probabilities'][0][0])
                else:
                    dialect_score = 0.5

            processing_time = time.time() - start_time

            return {
                'text': text,
                'sentiment': sentiment_name,
                'sentiment_score': sentiment_score,
                'dialect': dialect_name,
                'dialect_score': dialect_score,
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Error in combined analysis: {e}")
            raise

    async def analyze_batch(
        self,
        texts: List[str],
        language: str = "fa"
    ) -> Dict[str, Any]:
        """
        Batch analysis of multiple texts

        Args:
            texts: List of texts
            language: Language code

        Returns:
            Batch analysis results
        """
        start_time = time.time()
        results = []

        try:
            for text in texts:
                result = await self.analyze_combined(text, language)
                results.append(result)

            total_processing_time = time.time() - start_time

            return {
                'results': results,
                'total_texts': len(texts),
                'total_processing_time': total_processing_time
            }

        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            raise


# Global service instance
analysis_service = AnalysisService()
