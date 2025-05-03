import pandas as pd
import numpy as np
import re
from hazm import Normalizer, word_tokenize
from sklearn.model_selection import train_test_split

# کلاس پردازش داده‌های فارسی
class PersianTextPreprocessor:
    def __init__(self):
        self.normalizer = Normalizer()
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # تبدیل متن به فرم استاندارد
        text = self.normalizer.normalize(text)
        
        # حذف کاراکترهای خاص و اعداد
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
        
        # حذف فاصله‌های اضافی
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        return word_tokenize(text)

# کلاس پردازش داده‌های انگلیسی
class EnglishTextPreprocessor:
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # تبدیل به حروف کوچک
        text = text.lower()
        
        # حذف کاراکترهای خاص و اعداد
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # حذف فاصله‌های اضافی
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

# تابع بارگذاری و پردازش داده‌های فارسی
def load_persian_data(file_path):
    """
    بارگذاری دیتاست فارسی از فایل
    
    پارامترها:
        file_path: مسیر فایل CSV یا Excel
        
    خروجی:
        DataFrame پردازش شده با ستون‌های 'text' و 'sentiment'
    """
    # بررسی پسوند فایل
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("فرمت فایل باید CSV یا Excel باشد")
    
    # بررسی ستون‌های مورد نیاز
    required_columns = ['text', 'sentiment']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"ستون {col} در دیتاست یافت نشد")
    
    # پردازش متن‌ها
    preprocessor = PersianTextPreprocessor()
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    
    # حذف ردیف‌های خالی
    df = df.dropna(subset=['cleaned_text'])
    df = df[df['cleaned_text'] != '']
    
    # تبدیل برچسب‌های احساس به اعداد
    sentiment_mapping = {'مثبت': 2, 'خنثی': 1, 'منفی': 0}
    df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)
    
    return df

# تابع بارگذاری و پردازش داده‌های انگلیسی
def load_english_data(file_path):
    """
    بارگذاری دیتاست انگلیسی از فایل
    
    پارامترها:
        file_path: مسیر فایل CSV یا Excel
        
    خروجی:
        DataFrame پردازش شده با ستون‌های 'text' و 'sentiment'
    """
    # بررسی پسوند فایل
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("فرمت فایل باید CSV یا Excel باشد")
    
    # بررسی ستون‌های مورد نیاز
    required_columns = ['text', 'sentiment']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"ستون {col} در دیتاست یافت نشد")
    
    # پردازش متن‌ها
    preprocessor = EnglishTextPreprocessor()
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    
    # حذف ردیف‌های خالی
    df = df.dropna(subset=['cleaned_text'])
    df = df[df['cleaned_text'] != '']
    
    # تبدیل برچسب‌های احساس به اعداد
    sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)
    
    return df

# تابع تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
def split_data(df, test_size=0.2, random_state=42):
    """
    تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
    
    پارامترها:
        df: DataFrame شامل داده‌های پردازش شده
        test_size: نسبت داده‌های آزمون (پیش‌فرض: 0.2)
        random_state: عدد سید برای تکرارپذیری (پیش‌فرض: 42)
        
    خروجی:
        چهار مجموعه: متن‌های آموزش، متن‌های آزمون، برچسب‌های آموزش، برچسب‌های آزمون
    """
    X = df['cleaned_text'].values
    y = df['sentiment_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# کلاس پردازش‌کننده داده‌های چندزبانه
class MultilingualDataProcessor:
    def __init__(self):
        self.persian_preprocessor = PersianTextPreprocessor()
        self.english_preprocessor = EnglishTextPreprocessor()
        
    def clean_text(self, text, language="auto"):
        """
        پاکسازی متن با توجه به زبان آن
        
        پارامترها:
            text: متن ورودی
            language: زبان متن (پیش‌فرض: "auto" برای تشخیص خودکار)
            
        خروجی:
            متن پاکسازی‌شده
        """
        if language == "auto":
            language = self.detect_language(text)
            
        if language == "fa":
            return self.persian_preprocessor.clean_text(text)
        else:
            return self.english_preprocessor.clean_text(text)
    
    def detect_language(self, text):
        """
        تشخیص زبان متن (ساده)
        
        پارامترها:
            text: متن ورودی
            
        خروجی:
            کد زبان تشخیص داده شده ("fa" یا "en")
        """
        # بررسی حروف فارسی
        persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        text_chars = set(text.lower())
        
        if any(char in persian_chars for char in text_chars):
            return "fa"
        
        return "en"
    
    def tokenize(self, text, language="auto"):
        """
        توکن‌سازی متن با توجه به زبان آن
        
        پارامترها:
            text: متن ورودی
            language: زبان متن (پیش‌فرض: "auto" برای تشخیص خودکار)
            
        خروجی:
            لیستی از توکن‌ها
        """
        if language == "auto":
            language = self.detect_language(text)
            
        if language == "fa":
            return self.persian_preprocessor.tokenize(text)
        else:
            return text.split()

# تابع بارگذاری و پردازش داده‌های چندزبانه
def load_multilingual_data(file_paths, languages=None):
    """
    بارگذاری دیتاست‌های چندزبانه از فایل‌ها
    
    پارامترها:
        file_paths: لیستی از مسیرهای فایل CSV یا Excel
        languages: لیستی از کدهای زبان متناظر با فایل‌ها
        
    خروجی:
        DataFrame ترکیب‌شده با ستون‌های 'text', 'sentiment', 'language'
    """
    if languages is None:
        languages = ["auto"] * len(file_paths)
    
    if len(file_paths) != len(languages):
        raise ValueError("تعداد فایل‌ها و زبان‌ها باید یکسان باشد")
    
    dfs = []
    processor = MultilingualDataProcessor()
    
    for i, file_path in enumerate(file_paths):
        language = languages[i]
        
        # بارگذاری فایل
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"فرمت فایل {file_path} باید CSV یا Excel باشد")
        
        # بررسی ستون‌های مورد نیاز
        required_columns = ['text', 'sentiment']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون {col} در دیتاست {file_path} یافت نشد")
        
        # تشخیص زبان در صورت نیاز
        if language == "auto":
            df['language'] = df['text'].apply(processor.detect_language)
        else:
            df['language'] = language
        
        # پردازش متن‌ها
        df['cleaned_text'] = df.apply(
            lambda row: processor.clean_text(row['text'], row['language']),
            axis=1
        )
        
        # حذف ردیف‌های خالی
        df = df.dropna(subset=['cleaned_text'])
        df = df[df['cleaned_text'] != '']
        
        # استاندارد‌سازی برچسب‌های احساس
        df['sentiment_standard'] = df.apply(
            lambda row: standardize_sentiment(row['sentiment'], row['language']),
            axis=1
        )
        
        dfs.append(df)
    
    # ترکیب DataFrame‌ها
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

# تابع استاندارد‌سازی برچسب‌های احساس
def standardize_sentiment(sentiment, language):
    """
    استاندارد‌سازی برچسب‌های احساس به فرمت عددی
    
    پارامترها:
        sentiment: برچسب احساس
        language: زبان متن
        
    خروجی:
        برچسب استاندارد‌شده (0: منفی، 1: خنثی، 2: مثبت)
    """
    if language == "fa":
        sentiment_mapping = {'مثبت': 2, 'خنثی': 1, 'منفی': 0}
    else:
        sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    
    # تبدیل به حروف کوچک برای زبان‌های لاتین
    if language != "fa" and isinstance(sentiment, str):
        sentiment = sentiment.lower()
    
    return sentiment_mapping.get(sentiment, 1)  # پیش‌فرض: خنثی

# نمونه استفاده
if __name__ == "__main__":
    # مسیر فایل‌های داده (باید جایگزین شود)
    persian_data_path = "data/persian_sentiment_data.csv"
    english_data_path = "data/english_sentiment_data.csv"
    
    try:
        # بارگذاری داده‌های فارسی
        persian_df = load_persian_data(persian_data_path)
        print(f"تعداد نمونه‌های فارسی: {len(persian_df)}")
        
        # بارگذاری داده‌های انگلیسی
        english_df = load_english_data(english_data_path)
        print(f"تعداد نمونه‌های انگلیسی: {len(english_df)}")
        
        # تقسیم داده‌ها
        X_train_persian, X_test_persian, y_train_persian, y_test_persian = split_data(persian_df)
        X_train_english, X_test_english, y_train_english, y_test_english = split_data(english_df)
        
        # ذخیره‌سازی داده‌ها برای استفاده بعدی
        pd.DataFrame({
            'text': X_train_persian,
            'sentiment': y_train_persian
        }).to_csv("data/train_persian.csv", index=False)
        
        pd.DataFrame({
            'text': X_test_persian,
            'sentiment': y_test_persian
        }).to_csv("data/test_persian.csv", index=False)
        
        pd.DataFrame({
            'text': X_train_english,
            'sentiment': y_train_english
        }).to_csv("data/train_english.csv", index=False)
        
        pd.DataFrame({
            'text': X_test_english,
            'sentiment': y_test_english
        }).to_csv("data/test_english.csv", index=False)
        
        print("داده‌ها با موفقیت پردازش و ذخیره شدند.")
        
        # بارگذاری و پردازش داده‌های چندزبانه
        file_paths = [persian_data_path, english_data_path]
        languages = ["fa", "en"]
        
        multilingual_df = load_multilingual_data(file_paths, languages)
        print(f"تعداد نمونه‌های چندزبانه: {len(multilingual_df)}")
        
        # توزیع زبان‌ها
        language_distribution = multilingual_df['language'].value_counts()
        print("توزیع زبان‌ها:")
        print(language_distribution)
        
        # توزیع احساسات
        sentiment_distribution = multilingual_df['sentiment_standard'].value_counts()
        print("توزیع احساسات:")
        print(sentiment_distribution)
        
    except Exception as e:
        print(f"خطا در پردازش داده‌ها: {str(e)}")