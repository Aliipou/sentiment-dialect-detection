"""
Base classes for ML models
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """Base PyTorch Dataset class"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BaseModel(ABC):
    """Abstract base class for all models"""

    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Make predictions"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass

    def _ensure_list(self, texts: Union[str, List[str]]) -> List[str]:
        """Convert single string to list"""
        if isinstance(texts, str):
            return [texts]
        return texts
