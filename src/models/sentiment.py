"""
Sentiment analysis model for Persian text
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from .base import BaseModel, BaseDataset
from ..utils.config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class PersianSentimentAnalyzer(BaseModel):
    """Persian sentiment analysis using ParsBERT"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        num_labels: int = 3,
        device: Optional[str] = None
    ):
        """
        Initialize Persian sentiment analyzer

        Args:
            model_name: Pretrained model name
            num_labels: Number of sentiment labels
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name or settings.PERSIAN_MODEL_NAME
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu")
        self.tokenizer = None
        self.model = None

        logger.info(f"Initialized Persian sentiment analyzer with {self.model_name}")
        logger.info(f"Using device: {self.device}")

    def load_tokenizer(self):
        """Load tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully")
            return self.tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

    def load_model(self):
        """Load pretrained model"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            self.model.to(self.device)
            logger.info("Model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def create_datasets(
        self,
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        max_length: Optional[int] = None
    ):
        """Create PyTorch datasets"""
        if self.tokenizer is None:
            self.load_tokenizer()

        max_length = max_length or settings.MAX_LENGTH

        train_dataset = BaseDataset(
            train_texts, train_labels, self.tokenizer, max_length
        )
        test_dataset = BaseDataset(
            test_texts, test_labels, self.tokenizer, max_length
        )

        logger.info(f"Created datasets: {len(train_dataset)} train, {len(test_dataset)} test")
        return train_dataset, test_dataset

    def train(
        self,
        train_dataset,
        test_dataset,
        output_dir: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Train the sentiment model

        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            output_dir: Directory to save model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate

        Returns:
            Evaluation results
        """
        if self.model is None:
            self.load_model()

        output_dir = output_dir or settings.SENTIMENT_MODEL_PATH
        epochs = epochs or settings.NUM_EPOCHS
        batch_size = batch_size or settings.BATCH_SIZE
        learning_rate = learning_rate or settings.LEARNING_RATE

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=settings.WARMUP_STEPS,
            weight_decay=settings.WEIGHT_DECAY,
            logging_dir=str(settings.LOGS_DIR),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to=[]  # Disable wandb/tensorboard
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_result}")

        logger.info(f"Saving model to {output_dir}")
        self.save(output_dir)

        return eval_result

    def predict(
        self,
        texts: Union[str, List[str]],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Predict sentiment for texts

        Args:
            texts: Input text(s)
            return_probabilities: Whether to return probabilities

        Returns:
            Dictionary with predictions
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")

        texts = self._ensure_list(texts)

        try:
            encoded_inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=settings.MAX_LENGTH,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded_inputs)

            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            result = {
                'labels': predictions.cpu().numpy().tolist(),
                'probabilities': probabilities.cpu().numpy().tolist() if return_probabilities else None
            }

            return result

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def save(self, path: str) -> None:
        """Save model and tokenizer"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load(self, path: str) -> None:
        """Load trained model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.to(self.device)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


# Sentiment label mappings
SENTIMENT_LABELS_FA = {0: "منفی", 1: "خنثی", 2: "مثبت"}
SENTIMENT_LABELS_EN = {0: "negative", 1: "neutral", 2: "positive"}
