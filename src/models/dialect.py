"""
Persian dialect detection model
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Make hazm optional
try:
    from hazm import Normalizer
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False
    Normalizer = None

from .base import BaseModel
from ..utils.config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PersianDialectDetector(BaseModel):
    """Persian dialect detection using ML and rule-based methods"""

    # Supported dialects
    DIALECTS = ['تهرانی', 'اصفهانی', 'شیرازی', 'مشهدی', 'سایر']

    # Dialect keywords for rule-based detection
    DIALECT_KEYWORDS = {
        'تهرانی': ['داداش', 'مادر', 'دمت گرم', 'ایول', 'جون', 'مشتی', 'بابا', 'رفیق'],
        'اصفهانی': ['زِدی', 'خِدا', 'بَرِکَت', 'چِقَد', 'دِس', 'شُد', 'راس', 'زده'],
        'شیرازی': ['پَ', 'مَنُم', 'اُنجا', 'ما', 'چرا'],
        'مشهدی': ['آغو', 'بچه‌های مشد', 'شوما', 'خاک'],
        'سایر': ['خواهشمندم', 'اینجانب', 'فرمایید', 'مذکور', 'محترم']
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize dialect detector

        Args:
            model_path: Path to saved model
        """
        self.model_path = model_path or settings.DIALECT_MODEL_PATH
        self.model = None

        if HAZM_AVAILABLE:
            self.normalizer = Normalizer()
            logger.info("Initialized Persian dialect detector with hazm")
        else:
            self.normalizer = None
            logger.warning("Initialized Persian dialect detector without hazm (fallback mode)")

    def _fallback_normalize(self, text: str) -> str:
        """Fallback normalization without hazm"""
        replacements = {
            'ك': 'ک', 'ي': 'ی', 'ى': 'ی',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def load_data(self, file_path: str) -> tuple:
        """
        Load dialect data from CSV

        Args:
            file_path: Path to CSV file

        Returns:
            Tuple of (texts, labels)
        """
        try:
            df = pd.read_csv(file_path)

            if 'text' not in df.columns or 'dialect' not in df.columns:
                raise ValueError("Dataset must have 'text' and 'dialect' columns")

            # Filter valid dialects
            df = df[df['dialect'].isin(self.DIALECTS)]

            logger.info(f"Loaded {len(df)} samples")
            return df['text'].astype(str), df['dialect']

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def train(
        self,
        data_file: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train dialect detection model

        Args:
            data_file: Path to training data
            test_size: Test set ratio
            random_state: Random seed

        Returns:
            Training results
        """
        data_file = data_file or str(settings.DATA_DIR / "persian_dialect_data.csv")

        if not Path(data_file).exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load data
        X, y = self.load_data(data_file)

        # Normalize texts
        if HAZM_AVAILABLE and self.normalizer:
            X = X.apply(lambda text: self.normalizer.normalize(text))
        else:
            X = X.apply(lambda text: self._fallback_normalize(text))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),
            ('clf', MultinomialNB())
        ])

        # Train
        logger.info("Training model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))

        # Save model
        self.save(self.model_path)

        return {
            'accuracy': report['accuracy'],
            'report': report
        }

    def predict(
        self,
        texts: Union[str, List[str]],
        use_rules: bool = True
    ) -> Dict[str, Any]:
        """
        Predict dialect for texts

        Args:
            texts: Input text(s)
            use_rules: Whether to use rule-based fallback

        Returns:
            Predictions dictionary
        """
        texts = self._ensure_list(texts)
        results = []

        for text in texts:
            if HAZM_AVAILABLE and self.normalizer:
                normalized_text = self.normalizer.normalize(text)
            else:
                normalized_text = self._fallback_normalize(text)

            if self.model is not None:
                # Use ML model
                try:
                    predicted = self.model.predict([normalized_text])[0]
                    probabilities = self.model.predict_proba([normalized_text])[0]
                    label_index = self.DIALECTS.index(predicted)

                    results.append({
                        'dialect': predicted,
                        'label': label_index,
                        'confidence': float(probabilities[label_index])
                    })
                except Exception as e:
                    logger.error(f"Error in ML prediction: {e}")
                    if use_rules:
                        results.append(self._rule_based_predict(text))
                    else:
                        raise
            else:
                # Use rule-based method
                if use_rules:
                    results.append(self._rule_based_predict(text))
                else:
                    raise ValueError("Model not loaded and rule-based detection disabled")

        return {
            'labels': [r['label'] for r in results],
            'dialect_names': [r['dialect'] for r in results],
            'probabilities': [[r['confidence']] for r in results]
        }

    def _rule_based_predict(self, text: str) -> Dict[str, Any]:
        """
        Rule-based dialect prediction using keywords

        Args:
            text: Input text

        Returns:
            Prediction dictionary
        """
        if HAZM_AVAILABLE and self.normalizer:
            normalized_text = self.normalizer.normalize(text)
        else:
            normalized_text = self._fallback_normalize(text)
        scores = {}

        # Score each dialect based on keyword matches
        for dialect, keywords in self.DIALECT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in normalized_text)
            scores[dialect] = score

        # Find best match
        if max(scores.values()) == 0:
            predicted_dialect = 'سایر'
            confidence = 0.5
        else:
            predicted_dialect = max(scores, key=scores.get)
            total_score = sum(scores.values())
            confidence = scores[predicted_dialect] / total_score if total_score > 0 else 0

        return {
            'dialect': predicted_dialect,
            'label': self.DIALECTS.index(predicted_dialect),
            'confidence': confidence
        }

    def save(self, path: str) -> None:
        """Save model to disk"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load(self, path: str) -> None:
        """Load model from disk"""
        try:
            if Path(path).exists():
                self.model = joblib.load(path)
                logger.info(f"Model loaded from {path}")
            else:
                logger.warning(f"Model file not found: {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
