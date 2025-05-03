from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import torch
import logging

# وارد کردن کلاس‌های مدل
from sentiment_model import MultilingualSentimentAnalyzer
from dialect_detector import PersianDialectDetector, PersianTextPreprocessor
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

@app.on_event("startup")
async def startup_event():
    global sentiment_analyzer, dialect_detector, multilingual_analyzer
    
    logger.info("بارگذاری مدل‌ها...")
    
    try:
        # بارگذاری تنظیمات API
        import json
        import os
        
        config_path = "api_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {
                "sentiment_model_path": "models/sentiment_model",
                "dialect_model_path": "models/dialect_model",
                "multilingual_model_path": "models/multilingual_model"
            }
            
        # بارگذاری مدل تشخیص احساسات فارسی
        sentiment_analyzer = MultilingualSentimentAnalyzer()
        
        # تلاش برای بارگذاری مدل آموزش‌دیده، در صورت شکست بارگذاری مدل پیش‌فرض
        try:
            sentiment_analyzer.load_trained_model(config["sentiment_model_path"])
            logger.info("مدل آموزش‌دیده تشخیص احساسات فارسی با موفقیت بارگذاری شد.")
        except:
            logger.warning("مدل آموزش‌دیده تشخیص احساسات فارسی یافت نشد. بارگذاری مدل پیش‌فرض...")
            sentiment_analyzer.load_tokenizer()
            sentiment_analyzer.load_model()
        
        # بارگذاری مدل تشخیص لهجه فارسی
        dialect_detector = PersianDialectDetector()
        
        # تلاش برای بارگذاری مدل آموزش‌دیده، در صورت شکست استفاده از روش مبتنی بر قاعده
        try:
            dialect_detector.load_trained_model(config["dialect_model_path"])
            logger.info("مدل آموزش‌دیده تشخیص لهجه با موفقیت بارگذاری شد.")
        except:
            logger.warning("مدل آموزش‌دیده تشخیص لهجه یافت نشد. استفاده از روش مبتنی بر قاعده.")
            dialect_detector.load_tokenizer()
        
        # بارگذاری مدل چندزبانه
        multilingual_analyzer = AdvancedMultilingualSentimentAnalyzer()
        
        # تلاش برای بارگذاری مدل آموزش‌دیده، در صورت شکست بارگذاری مدل پیش‌فرض
        try:
            multilingual_analyzer.load_trained_model(config["multilingual_model_path"])
            logger.info("مدل آموزش‌دیده تشخیص احساسات چندزبانه با موفقیت بارگذاری شد.")
        except:
            logger.warning("مدل آموزش‌دیده تشخیص احساسات چندزبانه یافت نشد. بارگذاری مدل پیش‌فرض...")
            multilingual_analyzer.load_tokenizer()
            multilingual_analyzer.load_model()
    
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
        if hasattr(dialect_detector, 'model') and dialect_detector.model is not None:
            # استفاده از مدل آموزش‌دیده
            prediction = dialect_detector.predict(input_data.text)
            dialect_label = prediction["labels"][0]
            dialect_name = prediction["dialect_names"][0]
            confidence_score = float(prediction["probabilities"][0][dialect_label])
        else:
            # استفاده از روش مبتنی بر قاعده
            prediction = dialect_detector.get_rule_based_prediction(input_data.text)
            dialect_label = prediction["label"]
            dialect_name = prediction["dialect_name"]
            confidence_score = float(prediction["confidence"])
        
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
        # انتخاب آنالایزر مناسب بر اساس زبان
        if input_data.language == "fa":
            analyzer = sentiment_analyzer
        else:
            analyzer = multilingual_analyzer
            
        # تشخیص احساس
        sentiment_prediction = analyzer.predict(input_data.text)
        sentiment_label = sentiment_prediction["labels"][0]
        
        if input_data.language == "fa":
            sentiment_names = {0: "منفی", 1: "خنثی", 2: "مثبت"}
        else:
            lang = input_data.language
            sentiment_names = multilingual_analyzer.sentiment_labels.get(lang, multilingual_analyzer.sentiment_labels["en"])
            
        sentiment_name = sentiment_names.get(sentiment_label, "نامشخص")
        sentiment_score = float(sentiment_prediction["probabilities"][0][sentiment_label])
        
        response_data = {
            "text": input_data.text,
            "sentiment": sentiment_name,
            "sentiment_score": sentiment_score
        }
        
        # تشخیص لهجه (فقط برای فارسی)
        if input_data.language == "fa":
            if hasattr(dialect_detector, 'model') and dialect_detector.model is not None:
                # استفاده از مدل آموزش‌دیده
                dialect_prediction = dialect_detector.predict(input_data.text)
                dialect_label = dialect_prediction["labels"][0]
                dialect_name = dialect_prediction["dialect_names"][0]
                dialect_score = float(dialect_prediction["probabilities"][0][dialect_label])
            else:
                # استفاده از روش مبتنی بر قاعده
                dialect_prediction = dialect_detector.get_rule_based_prediction(input_data.text)
                dialect_name = dialect_prediction["dialect_name"]
                dialect_score = float(dialect_prediction["confidence"])
                
            response_data["dialect"] = dialect_name
            response_data["dialect_score"] = dialect_score
        
        # زمان پردازش
        processing_time = time.time() - start_time
        response_data["processing_time"] = processing_time
        
        return CombinedResponse(**response_data)
    
    except Exception as e:
        logger.error(f"خطا در تحلیل متن: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل متن: {str(e)}")

@app.post("/batch")
async def batch_analyze(input_data: BatchTextInput):
    if not input_data.texts:
        raise HTTPException(status_code=400, detail="لیست متن‌های ورودی خالی است")
    
    results = []
    start_time = time.time()
    
    try:
        for text in input_data.texts:
            single_input = TextInput(text=text, language=input_data.language)
            result = await analyze_text(single_input)
            results.append(result)
        
        total_processing_time = time.time() - start_time
        
        return {
            "results": results,
            "total_texts": len(results),
            "total_processing_time": total_processing_time
        }
    
    except Exception as e:
        logger.error(f"خطا در پردازش گروهی: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در پردازش گروهی: {str(e)}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"مسیر: {request.url.path} - زمان: {process_time:.4f}s - وضعیت: {response.status_code}")
    
    return response

# اجرای برنامه با uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sentiment_api:app", host="0.0.0.0", port=8000, reload=True)