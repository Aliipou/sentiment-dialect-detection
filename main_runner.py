import argparse
import subprocess
import os
import sys
import time
import logging

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """بررسی نصب وابستگی‌ها"""
    try:
        import torch
        import transformers
        import fastapi
        import streamlit
        import pandas
        import numpy
        import sklearn
        import hazm
        logger.info("✓ تمام وابستگی‌های اصلی نصب شده‌اند")
        return True
    except ImportError as e:
        logger.error(f"✗ وابستگی نصب نشده: {e}")
        return False

def install_dependencies():
    """نصب وابستگی‌های مورد نیاز"""
    logger.info("در حال نصب وابستگی‌ها...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_directories():
    """ایجاد پوشه‌های مورد نیاز"""
    directories = ['models', 'models/sentiment_model', 'models/dialect_model', 'models/multilingual_model', 'data', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("✓ پوشه‌های مورد نیاز ایجاد شدند")

def run_data_preprocessing():
    """اجرای پیش‌پردازش داده‌ها"""
    logger.info("در حال پیش‌پردازش داده‌ها...")
    try:
        subprocess.run([sys.executable, "data_preprocessing.py"], check=True)
        logger.info("✓ پیش‌پردازش داده‌ها با موفقیت انجام شد")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ خطا در پیش‌پردازش داده‌ها: {e}")
        raise

def train_models():
    """آموزش مدل‌ها"""
    logger.info("در حال آموزش مدل‌ها...")
    
    # آموزش مدل احساسات فارسی
    try:
        logger.info("آموزش مدل احساسات فارسی...")
        subprocess.run([sys.executable, "-c", """
import sentiment_model
from data_preprocessing import load_persian_data, split_data

# بارگذاری و آماده‌سازی داده‌ها
persian_df = load_persian_data('data/persian_sentiment_data.csv')
X_train, X_test, y_train, y_test = split_data(persian_df)

# آموزش مدل
analyzer = sentiment_model.MultilingualSentimentAnalyzer()
tokenizer = analyzer.load_tokenizer()
model = analyzer.load_model()

train_dataset, test_dataset = analyzer.create_datasets(
    X_train, y_train, X_test, y_test
)

results = analyzer.train(train_dataset, test_dataset, output_dir='models/sentiment_model')
print(f'نتایج آموزش: {results}')
"""], check=True)
        logger.info("✓ مدل احساسات فارسی آموزش داده شد")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ خطا در آموزش مدل احساسات: {e}")
    
    # آموزش مدل لهجه
    try:
        logger.info("آموزش مدل لهجه...")
        subprocess.run([sys.executable, "-c", """
import dialect_detector

detector = dialect_detector.PersianDialectDetector(model_path='models/dialect_model.joblib')
detector.train_model('data/persian_dialect_data.csv')
"""], check=True)
        logger.info("✓ مدل لهجه آموزش داده شد")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ خطا در آموزش مدل لهجه: {e}")
    
    # آموزش مدل چندزبانه
    try:
        logger.info("آموزش مدل چندزبانه...")
        subprocess.run([sys.executable, "-c", """
import multilingual_model
from data_preprocessing import load_multilingual_data

# مسیرهای داده‌ها
file_paths = ['data/persian_sentiment_data.csv', 'data/english_sentiment_data.csv']
languages = ['fa', 'en']

# بارگذاری داده‌های چندزبانه
multilingual_df = load_multilingual_data(file_paths, languages)

# تقسیم داده‌ها
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    multilingual_df['cleaned_text'].values,
    multilingual_df['sentiment_standard'].values,
    test_size=0.2,
    random_state=42,
    stratify=multilingual_df['sentiment_standard'].values
)

# آموزش مدل
analyzer = multilingual_model.AdvancedMultilingualSentimentAnalyzer()
tokenizer = analyzer.load_tokenizer()
model = analyzer.load_model()

train_dataset, test_dataset = analyzer.create_datasets(
    X_train, y_train, X_test, y_test
)

results = analyzer.train(train_dataset, test_dataset, output_dir='models/multilingual_model')
print(f'نتایج آموزش مدل چندزبانه: {results}')
"""], check=True)
        logger.info("✓ مدل چندزبانه آموزش داده شد")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ خطا در آموزش مدل چندزبانه: {e}")

def run_api():
    """راه‌اندازی API"""
    logger.info("در حال راه‌اندازی API...")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "sentiment_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

def run_streamlit():
    """راه‌اندازی Streamlit"""
    logger.info("در حال راه‌اندازی رابط کاربری...")
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

def main():
    parser = argparse.ArgumentParser(description='اجرای پروژه تشخیص احساسات و لهجه')
    parser.add_argument('--skip-setup', action='store_true', help='رد کردن نصب کتابخانه‌ها')
    parser.add_argument('--skip-training', action='store_true', help='رد کردن آموزش مدل‌ها')
    parser.add_argument('--api-only', action='store_true', help='فقط اجرای API')
    parser.add_argument('--ui-only', action='store_true', help='فقط اجرای رابط کاربری')
    
    args = parser.parse_args()
    
    logger.info("=== شروع راه‌اندازی پروژه ===")
    
    # نصب وابستگی‌ها
    if not args.skip_setup:
        if not check_dependencies():
            install_dependencies()
    
    # ایجاد پوشه‌های مورد نیاز
    create_directories()
    
    # پیش‌پردازش و آموزش
    if not args.skip_training:
        try:
            run_data_preprocessing()
            train_models()
        except Exception as e:
            logger.error(f"خطا در آموزش مدل‌ها: {e}")
            if input("آیا می‌خواهید بدون آموزش مدل‌ها ادامه دهید؟ (y/n): ").lower() != 'y':
                sys.exit(1)
    
    # اجرای سرویس‌ها
    api_process = None
    streamlit_process = None
    
    try:
        if not args.ui_only:
            api_process = run_api()
            time.sleep(5)  # صبر برای راه‌اندازی API
        
        if not args.api_only:
            streamlit_process = run_streamlit()
        
        logger.info("=== پروژه با موفقیت راه‌اندازی شد ===")
        logger.info("API: http://localhost:8000")
        logger.info("UI: http://localhost:8501")
        logger.info("برای خروج Ctrl+C را فشار دهید")
        
        # نگه داشتن برنامه
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\nدر حال بستن پروژه...")
        if api_process:
            api_process.terminate()
        if streamlit_process:
            streamlit_process.terminate()
        logger.info("پروژه بسته شد.")
    except Exception as e:
        logger.error(f"خطای غیرمنتظره: {e}")
        if api_process:
            api_process.terminate()
        if streamlit_process:
            streamlit_process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()