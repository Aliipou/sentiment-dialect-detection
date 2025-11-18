from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import torch
import logging
import os
import json

# وارد کردن کلاس‌های مدل
from sentiment_model import MultilingualSentimentAnalyzer
from dialect_detector import PersianDialectDetector
from multilingual_model import AdvancedMultilingualSentimentAnalyzer

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ایجاد نمونه از FastAPI
app = FastAPI(
    title="API تشخیص احساسات و لهجه متن چندزبانه",
    description="API برای تشخیص احساسات در متن‌های فارسی و انگلیسی و تشخیص لهجه در متن‌های فارسی",
    version="1.0.0"
)

# تنظیمات CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تعریف مدل‌ها برای API
class TextInput(BaseModel):
    text: str
    language: Optional[str] = "fa"

class BatchTextInput(BaseModel):
    texts: List[str]
    language: Optional[str] = "fa"

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    sentiment_score: float
    processing_time: float

class DialectResponse(BaseModel):
    text: str
    dialect: str
    dialect_score: float
    processing_time: float

class CombinedResponse(BaseModel):
    text: str
    sentiment: str
    sentiment_score: float
    dialect: Optional[str] = None
    dialect_score: Optional[float] = None
    processing_time: float

# بارگذاری مدل‌ها
sentiment_analyzer = None
dialect_detector = None
multilingual_analyzer = None

def load_config():
    """بارگذاری تنظیمات API"""
    config_path = "api_config.json"
    default_config = {
        "sentiment_model_path": "models/sentiment_model",
        "dialect_model_path": "models/dialect_model.joblib",
        "multilingual_model_path": "models/multilingual_model"
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                # ترکیب با تنظیمات پیش‌فرض
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            logger.error(f"خطا در خواندن فایل تنظیمات: {e}")
            return default_config
    else:
        # ایجاد فایل تنظیمات پیش‌فرض
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=4)
        return default_config

@app.on_event("startup")
async def startup_event():
    global sentiment_analyzer, dialect_detector, multilingual_analyzer
    
    logger.info("بارگذاری مدل‌ها...")
    
    try:
        # بارگذاری تنظیمات
        config = load_config()
        
        # بارگذاری مدل تشخیص احساسات فارسی
        sentiment_analyzer = MultilingualSentimentAnalyzer()
        
        try:
            if os.path.exists(config["sentiment_model_path"]):
                sentiment_analyzer.load_trained_model(config["sentiment_model_path"])
                logger.info("مدل آموزش‌دیده تشخیص احساسات فارسی با موفقیت بارگذاری شد.")
            else:
                logger.warning("مدل آموزش‌دیده تشخیص احساسات فارسی یافت نشد. بارگذاری مدل پیش‌فرض...")
                sentiment_analyzer.load_tokenizer()
                sentiment_analyzer.load_model()
        except Exception as e:
            logger.warning(f"خطا در بارگذاری مدل آموزش‌دیده: {e}")
            sentiment_analyzer.load_tokenizer()
            sentiment_analyzer.load_model()
        
        # بارگذاری مدل تشخیص لهجه فارسی
        dialect_detector = PersianDialectDetector(model_path=config["dialect_model_path"])
        
        try:
            if not dialect_detector.load_model():
                logger.warning("مدل آموزش‌دیده تشخیص لهجه یافت نشد. استفاده از روش مبتنی بر قاعده.")
        except Exception as e:
            logger.warning(f"خطا در بارگذاری مدل لهجه: {e}")
        
        # بارگذاری مدل چندزبانه
        multilingual_analyzer = AdvancedMultilingualSentimentAnalyzer()
        
        try:
            if os.path.exists(config["multilingual_model_path"]):
                multilingual_analyzer.load_trained_model(config["multilingual_model_path"])
                logger.info("مدل آموزش‌دیده تشخیص احساسات چندزبانه با موفقیت بارگذاری شد.")
            else:
                logger.warning("مدل آموزش‌دیده تشخیص احساسات چندزبانه یافت نشد. بارگذاری مدل پیش‌فرض...")
                multilingual_analyzer.load_tokenizer()
                multilingual_analyzer.load_model()
        except Exception as e:
            logger.warning(f"خطا در بارگذاری مدل چندزبانه: {e}")
            multilingual_analyzer.load_tokenizer()
            multilingual_analyzer.load_model()
        
        logger.info("همه مدل‌ها با موفقیت بارگذاری شدند.")
    
    except Exception as e:
        logger.error(f"خطا در بارگذاری مدل‌ها: {str(e)}")
        raise

@app.get("/")
def read_root():
    return {
        "message": "API تشخیص احساسات و لهجه متن چندزبانه",
        "version": "1.0.0",
        "endpoints": [
            "/sentiment - تشخیص احساسات متن",
            "/dialect - تشخیص لهجه متن فارسی",
            "/analyze - تحلیل کامل متن (احساسات و لهجه)",
            "/batch - پردازش گروهی متن‌ها"
        ]
    }

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    start_time = time.time()
    
    if not input_data.text:
        raise HTTPException(status_code=400, detail="متن ورودی خالی است")
    
    try:
        # انتخاب آنالایزر مناسب بر اساس زبان
        if input_data.language == "fa":
            analyzer = sentiment_analyzer
        else:
            analyzer = multilingual_analyzer
        
        # تشخیص احساس
        prediction = analyzer.predict(input_data.text)
        
        # تبدیل برچسب عددی به متنی
        sentiment_label = prediction["labels"][0]
        
        if input_data.language == "fa":
            sentiment_names = {0: "منفی", 1: "خنثی", 2: "مثبت"}
        else:
            lang = input_data.language
            sentiment_names = multilingual_analyzer.sentiment_labels.get(lang, multilingual_analyzer.sentiment_labels["en"])
            
        sentiment_name = sentiment_names.get(sentiment_label, "نامشخص")
        
        # محاسبه امتیاز اطمینان
        confidence_score = float(prediction["probabilities"][0][sentiment_label])
        
        # زمان پردازش
        processing_time = time.time() - start_time
        
        return SentimentResponse(
            text=input_data.text,
            sentiment=sentiment_name,
            sentiment_score=confidence_score,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"خطا در تشخیص احساس: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در تشخیص احساس: {str(e)}")

@app.post("/dialect", response_model=DialectResponse)
async def analyze_dialect(input_data: TextInput):
    start_time = time.time()
    
    if not input_data.text:
        raise HTTPException(status_code=400, detail="متن ورودی خالی است")
    
    if input_data.language != "fa":
        raise HTTPException(status_code=400, detail="تشخیص لهجه فقط برای زبان فارسی پشتیبانی می‌شود")
    
    try:
        # تشخیص لهجه
        prediction = dialect_detector.predict_dialect(input_data.text)
        
        dialect_label = prediction["labels"][0]
        dialect_name = prediction["dialect_names"][0]
        
        # محاسبه امتیاز اطمینان
        if len(prediction["probabilities"][0]) > dialect_label:
            confidence_score = float(prediction["probabilities"][0][dialect_label])
        else:
            confidence_score = float(prediction["probabilities"][0][0])
        
        # زمان پردازش
        processing_time = time.time() - start_time
        
        return DialectResponse(
            text=input_data.text,
            dialect=dialect_name,
            dialect_score=confidence_score,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"خطا در تشخیص لهجه: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در تشخیص لهجه: {str(e)}")

@app.post("/analyze", response_model=CombinedResponse)
async def analyze_text(input_data: TextInput):
    start_time = time.time()
    
    if not input_data.text:
        raise HTTPException(status_code=400, detail="متن ورودی خالی است")
    
    try:
        # انتخاب آنالایزر مناس