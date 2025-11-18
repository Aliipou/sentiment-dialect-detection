"""
Unit tests for data loading
"""
import pytest
import pandas as pd
from pathlib import Path
from src.preprocessing.data_loader import DataLoader


class TestDataLoader:
    """Tests for DataLoader class"""

    def test_load_data_csv(self, temp_data_dir):
        """Test loading CSV file"""
        # Create test CSV
        df = pd.DataFrame({
            'text': ['test1', 'test2'],
            'sentiment': ['positive', 'negative']
        })
        csv_path = temp_data_dir / "test.csv"
        df.to_csv(csv_path, index=False)

        loader = DataLoader()
        loaded_df = loader.load_data(csv_path)

        assert len(loaded_df) == 2
        assert 'text' in loaded_df.columns
        assert 'sentiment' in loaded_df.columns

    def test_load_data_file_not_found(self):
        """Test error when file not found"""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_data("nonexistent.csv")

    def test_load_data_invalid_format(self, temp_data_dir):
        """Test error with unsupported file format"""
        invalid_file = temp_data_dir / "test.txt"
        invalid_file.write_text("test")

        loader = DataLoader()
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_data(invalid_file)

    def test_load_data_missing_columns(self, temp_data_dir):
        """Test error when required columns missing"""
        df = pd.DataFrame({'wrong_column': ['test']})
        csv_path = temp_data_dir / "test.csv"
        df.to_csv(csv_path, index=False)

        loader = DataLoader()
        with pytest.raises(ValueError, match="not found in dataset"):
            loader.load_data(csv_path)

    def test_preprocess_persian_data(self, temp_data_dir):
        """Test Persian data preprocessing"""
        df = pd.DataFrame({
            'text': ['این خوب است', 'این بد است', 'این متوسط است'],
            'sentiment': ['مثبت', 'منفی', 'خنثی']
        })

        loader = DataLoader()
        processed_df = loader.preprocess_persian_data(df)

        assert 'cleaned_text' in processed_df.columns
        assert 'sentiment_label' in processed_df.columns
        assert len(processed_df) == 3
        assert processed_df['sentiment_label'].tolist() == [2, 0, 1]

    def test_preprocess_english_data(self):
        """Test English data preprocessing"""
        df = pd.DataFrame({
            'text': ['this is good', 'this is bad', 'this is okay'],
            'sentiment': ['positive', 'negative', 'neutral']
        })

        loader = DataLoader()
        processed_df = loader.preprocess_english_data(df)

        assert 'cleaned_text' in processed_df.columns
        assert 'sentiment_label' in processed_df.columns
        assert len(processed_df) == 3
        assert processed_df['sentiment_label'].tolist() == [2, 0, 1]

    def test_split_data(self):
        """Test data splitting"""
        df = pd.DataFrame({
            'cleaned_text': ['text1', 'text2', 'text3', 'text4'] * 5,
            'sentiment_label': [0, 1, 2, 0] * 5
        })

        loader = DataLoader()
        X_train, X_test, y_train, y_test = loader.split_data(df, test_size=0.2, random_state=42)

        assert len(X_train) == 16  # 80% of 20
        assert len(X_test) == 4    # 20% of 20
        assert len(y_train) == 16
        assert len(y_test) == 4
