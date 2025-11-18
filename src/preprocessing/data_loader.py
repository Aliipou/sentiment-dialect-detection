"""
Data loading and management utilities
"""
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split

from .text_processor import MultilingualTextPreprocessor, PersianTextPreprocessor, EnglishTextPreprocessor
from ..utils.config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Data loader for sentiment and dialect datasets"""

    def __init__(self):
        self.multilingual_processor = MultilingualTextPreprocessor()
        self.persian_processor = PersianTextPreprocessor()
        self.english_processor = EnglishTextPreprocessor()

    def load_data(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        label_column: str = "sentiment"
    ) -> pd.DataFrame:
        """
        Load data from CSV or Excel file

        Args:
            file_path: Path to data file
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If file format is unsupported or columns are missing
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load based on file extension
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            logger.info(f"Loaded {len(df)} rows from {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

        # Validate columns
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in dataset")

        return df

    def preprocess_persian_data(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "sentiment"
    ) -> pd.DataFrame:
        """
        Preprocess Persian sentiment data

        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Clean texts
        df['cleaned_text'] = df[text_column].apply(self.persian_processor.clean_text)

        # Remove empty rows
        df = df[df['cleaned_text'].str.strip() != '']
        df = df.dropna(subset=['cleaned_text'])

        # Map sentiment labels to numeric
        sentiment_mapping = {'مثبت': 2, 'خنثی': 1, 'منفی': 0}
        df['sentiment_label'] = df[label_column].map(sentiment_mapping)

        # Remove rows with unmapped labels
        df = df.dropna(subset=['sentiment_label'])
        df['sentiment_label'] = df['sentiment_label'].astype(int)

        logger.info(f"Preprocessed Persian data: {len(df)} rows remaining")
        return df

    def preprocess_english_data(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "sentiment"
    ) -> pd.DataFrame:
        """
        Preprocess English sentiment data

        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Clean texts
        df['cleaned_text'] = df[text_column].apply(self.english_processor.clean_text)

        # Remove empty rows
        df = df[df['cleaned_text'].str.strip() != '']
        df = df.dropna(subset=['cleaned_text'])

        # Map sentiment labels to numeric
        sentiment_mapping = {
            'positive': 2,
            'neutral': 1,
            'negative': 0
        }

        # Try lowercase mapping
        df['sentiment_label'] = df[label_column].str.lower().map(sentiment_mapping)

        # Remove rows with unmapped labels
        df = df.dropna(subset=['sentiment_label'])
        df['sentiment_label'] = df['sentiment_label'].astype(int)

        logger.info(f"Preprocessed English data: {len(df)} rows remaining")
        return df

    def preprocess_multilingual_data(
        self,
        file_paths: List[Union[str, Path]],
        languages: List[str]
    ) -> pd.DataFrame:
        """
        Load and preprocess multilingual data

        Args:
            file_paths: List of data file paths
            languages: List of language codes corresponding to files

        Returns:
            Combined preprocessed DataFrame
        """
        if len(file_paths) != len(languages):
            raise ValueError("Number of file paths must match number of languages")

        dfs = []

        for file_path, language in zip(file_paths, languages):
            df = self.load_data(file_path)

            # Preprocess based on language
            if language == "fa":
                df_processed = self.preprocess_persian_data(df)
            elif language == "en":
                df_processed = self.preprocess_english_data(df)
            else:
                logger.warning(f"Unknown language '{language}', skipping {file_path}")
                continue

            df_processed['language'] = language
            dfs.append(df_processed)

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined multilingual data: {len(combined_df)} total rows")

        return combined_df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train and test sets

        Args:
            df: Input DataFrame with 'cleaned_text' and 'sentiment_label'
            test_size: Test set ratio
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or settings.TEST_SIZE
        random_state = random_state or settings.RANDOM_STATE

        X = df['cleaned_text'].values
        y = df['sentiment_label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test


# Convenience functions
def load_persian_sentiment_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load and preprocess Persian sentiment data"""
    file_path = file_path or str(settings.DATA_DIR / "persian_sentiment_data.csv")
    loader = DataLoader()
    df = loader.load_data(file_path)
    return loader.preprocess_persian_data(df)


def load_english_sentiment_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load and preprocess English sentiment data"""
    file_path = file_path or str(settings.DATA_DIR / "english_sentiment_data.csv")
    loader = DataLoader()
    df = loader.load_data(file_path)
    return loader.preprocess_english_data(df)


def load_dialect_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load Persian dialect data"""
    file_path = file_path or str(settings.DATA_DIR / "persian_dialect_data.csv")
    loader = DataLoader()
    return loader.load_data(file_path, label_column="dialect")
