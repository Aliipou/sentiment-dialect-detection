import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import os
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple

# راه‌اندازی لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# تعریف کلاس دیتاست برای ترنسفورمر
class MultilingualSentimentDataset(Dataset):
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

# تابع ارزیابی مدل
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# کلاس اصلی مدل تشخیص احساسات چندزبانه
class AdvancedMultilingualSentimentAnalyzer:
    def __init__(self, model_name=None, num_labels=3):
        """
        راه‌اندازی مدل تشخیص احساسات چندزبانه
        
        پارامترها:
            model_name: نام مدل پیش‌آموزش‌دیده (پیش‌فرض: "xlm-roberta-base" برای پوشش چندزبانه)
            num_labels: تعداد برچسب‌های احساس (پیش‌فرض: 3 - مثبت، خنثی، منفی)
        """
        if model_name is None:
            # از XLM-RoBERTa برای پشتیبانی از بیش از 100 زبان استفاده می‌کنیم
            self.model_name = "xlm-roberta-base"
        else:
            self.model_name = model_name
            
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # دیکشنری برچسب‌های احساس به زبان‌های مختلف
        self.sentiment_labels = {
            "fa": {0: "منفی", 1: "خنثی", 2: "مثبت"},
            "en": {0: "negative", 1: "neutral", 2: "positive"},
            "fr": {0: "négatif", 1: "neutre", 2: "positif"},
            "de": {0: "negativ", 1: "neutral", 2: "positiv"},
            "es": {0: "negativo", 1: "neutral", 2: "positivo"},
            "it": {0: "negativo", 1: "neutrale", 2: "positivo"},
            "pt": {0: "negativo", 1: "neutro", 2: "positivo"},
            "nl": {0: "negatief", 1: "neutraal", 2: "positief"},
            "sv": {0: "negativ", 1: "neutral", 2: "positiv"},
            "no": {0: "negativ", 1: "nøytral", 2: "positiv"},
            "da": {0: "negativ", 1: "neutral", 2: "positiv"},
            "fi": {0: "negatiivinen", 1: "neutraali", 2: "positiivinen"},
            "el": {0: "αρνητικό", 1: "ουδέτερο", 2: "θετικό"},
            "ru": {0: "негативный", 1: "нейтральный", 2: "позитивный"},
            "pl": {0: "negatywny", 1: "neutralny", 2: "pozytywny"},
            "cs": {0: "negativní", 1: "neutrální", 2: "pozitivní"},
            "hu": {0: "negatív", 1: "semleges", 2: "pozitív"},
            "ro": {0: "negativ", 1: "neutru", 2: "pozitiv"},
            "tr": {0: "olumsuz", 1: "nötr", 2: "olumlu"}
        }
        
        # پیش‌فرض زبان انگلیسی
        self.default_lang = "en"
        
        logger.info(f"مدل چندزبانه با پشتیبانی از {len(self.sentiment_labels)} زبان راه‌اندازی شد.")
        logger.info(f"استفاده از دستگاه: {self.device}")
        
    def load_tokenizer(self):
        """بارگذاری توکنایزر مناسب برای مدل"""
        logger.info(f"بارگذاری توکنایزر از {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer
        
    def load_model(self):
        """بارگذاری مدل پیش‌آموزش‌دیده"""
        logger.info(f"بارگذاری مدل از {self.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        return self.model
    
    def create_datasets(self, train_texts, train_labels, test_texts, test_labels):
        """
        ایجاد دیتاست‌های آموزش و ارزیابی
        
        پارامترها:
            train_texts: متن‌های آموزش
            train_labels: برچسب‌های آموزش
            test_texts: متن‌های آزمون
            test_labels: برچسب‌های آزمون
            
        خروجی:
            دیتاست‌های آموزش و ارزیابی
        """
        if self.tokenizer is None:
            self.load_tokenizer()
            
        train_dataset = MultilingualSentimentDataset(train_texts, train_labels, self.tokenizer)
        test_dataset = MultilingualSentimentDataset(test_texts, test_labels, self.tokenizer)
        
        return train_dataset, test_dataset
    
    def train(self, train_dataset, test_dataset, output_dir="multilingual_sentiment_model", epochs=3, batch_size=16):
        """
        آموزش مدل
        
        پارامترها:
            train_dataset: دیتاست آموزش
            test_dataset: دیتاست ارزیابی
            output_dir: مسیر ذخیره‌سازی مدل
            epochs: تعداد دوره‌های آموزش
            batch_size: اندازه دسته
            
        خروجی:
            نتایج آموزش
        """
        if self.model is None:
            self.load_model()
            
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        logger.info("شروع آموزش مدل...")
        trainer.train()
        
        # ارزیابی نهایی
        logger.info("ارزیابی مدل...")
        eval_result = trainer.evaluate()
        logger.info(f"نتایج ارزیابی: {eval_result}")
        
        # ذخیره مدل نهایی
        logger.info(f"ذخیره مدل در {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return eval_result
    
    def predict(self, texts, language="en"):
        """
        پیش‌بینی احساس برای متن‌های جدید
        
        پارامترها:
            texts: لیستی از متن‌ها یا یک متن
            language: کد زبان متن (پیش‌فرض: "en")
            
        خروجی:
            دیکشنری شامل برچسب‌های پیش‌بینی‌شده و احتمالات
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("مدل یا توکنایزر بارگذاری نشده است")
            
        # تعیین نام‌های برچسب بر اساس زبان
        lang_labels = self.sentiment_labels.get(language, self.sentiment_labels[self.default_lang])
            
        if isinstance(texts, str):
            texts = [texts]
            
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        # تبدیل شاخص‌ها به نام‌های احساس با توجه به زبان
        sentiment_names = [lang_labels[label.item()] for label in predictions]
        
        result = {
            'labels': predictions.cpu().numpy(),
            'sentiment_names': sentiment_names,
            'probabilities': probabilities.cpu().numpy(),
            'language': language
        }
        
        return result
    
    def load_trained_model(self, model_path):
        """
        بارگذاری مدل آموزش‌دیده
        
        پارامترها:
            model_path: مسیر مدل ذخیره‌شده
        """
        logger.info(f"بارگذاری مدل آموزش‌دیده از {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        logger.info("مدل آموزش‌دیده با موفقیت بارگذاری شد.")
    
    def detect_language(self, text):
        """
        تشخیص زبان متن (ساده)
        
        پارامترها:
            text: متن ورودی
            
        خروجی:
            کد زبان تشخیص داده شده
        """
        # این یک تابع ساده برای تشخیص زبان است
        # در یک محیط واقعی، باید از کتابخانه‌های مخصوص مانند langdetect استفاده کنید
        
        # بررسی حروف فارسی
        persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        text_chars = set(text.lower())
        
        if any(char in persian_chars for char in text_chars):
            return "fa"
            
        # بررسی سایر زبان‌ها (تشخیص ساده)
        languages = {
            "fr": set("àâçéèêëîïôùûüÿæœ"),
            "de": set("äöüß"),
            "es": set("áéíóúüñ¿¡"),
            "it": set("àèéìíîòóùú"),
            "pt": set("áàâãçéêíóôõú"),
            "nl": set("ijëïöü"),
            "sv": set("åäö"),
            "no": set("åæø"),
            "da": set("åæø"),
            "fi": set("äöå"),
            "el": set("αβγδεζηθικλμνξοπρστυφχψω"),
            "ru": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
            "pl": set("ąćęłńóśźż"),
            "cs": set("áčďéěíňóřšťúůýž"),
            "hu": set("áéíóöőúüű"),
            "ro": set("ăâîșț"),
            "tr": set("çğıİöşü")
        }
        
        # محاسبه نسبت کاراکترهای خاص هر زبان در متن
        scores = {}
        for lang, chars in languages.items():
            if any(char in chars for char in text_chars):
                match_count = sum(1 for char in text if char in chars)
                scores[lang] = match_count / len(text) if len(text) > 0 else 0
        
        # اگر هیچ زبانی تشخیص داده نشد، پیش‌فرض انگلیسی
        if not scores:
            return "en"
            
        # انتخاب زبان با بیشترین امتیاز
        detected_lang = max(scores, key=scores.get) if scores else "en"
        return detected_lang
    
    def analyze_sentiment_with_language_detection(self, text):
        """
        تحلیل احساسات با تشخیص خودکار زبان
        
        پارامترها:
            text: متن ورودی
            
        خروجی:
            دیکشنری شامل نتایج تحلیل احساس و زبان تشخیص داده شده
        """
        # تشخیص زبان
        detected_lang = self.detect_language(text)
        
        # تحلیل احساس با توجه به زبان تشخیص داده شده
        prediction = self.predict(text, language=detected_lang)
        
        # افزودن زبان تشخیص داده شده به نتایج
        prediction['detected_language'] = detected_lang
        
        return prediction

# تابع کمکی برای آموزش مدل با داده‌های چندزبانه
def train_multilingual_model(data_paths, output_dir, model_name="xlm-roberta-base", epochs=3):
    """
    آموزش مدل چندزبانه با داده‌های مختلف
    
    پارامترها:
        data_paths: دیکشنری شامل مسیر داده‌های هر زبان
        output_dir: مسیر ذخیره مدل
        model_name: نام مدل پایه
        epochs: تعداد دوره‌های آموزش
    """
    # ایجاد نمونه از آنالایزر
    analyzer = AdvancedMultilingualSentimentAnalyzer(model_name=model_name)
    
    # بارگذاری توکنایزر و مدل
    tokenizer = analyzer.load_tokenizer()
    model = analyzer.load_model()
    
    # جمع‌آوری داده‌های همه زبان‌ها
    all_train_texts = []
    all_train_labels = []
    all_test_texts = []
    all_test_labels = []
    
    for lang, paths in data_paths.items():
        logger.info(f"بارگذاری داده‌های زبان {lang}...")
        
        # بارگذاری داده‌های آموزش
        train_df = pd.read_csv(paths['train'])
        
        # بارگذاری داده‌های آزمون
        test_df = pd.read_csv(paths['test'])
        
        # افزودن به لیست کلی
        all_train_texts.extend(train_df['text'].values)
        all_train_labels.extend(train_df['sentiment'].values)
        all_test_texts.extend(test_df['text'].values)
        all_test_labels.extend(test_df['sentiment'].values)
    
    logger.info(f"مجموع داده‌های آموزش: {len(all_train_texts)}")
    logger.info(f"مجموع داده‌های آزمون: {len(all_test_texts)}")
    
    # ایجاد دیتاست‌ها
    train_dataset, test_dataset = analyzer.create_datasets(
        all_train_texts, 
        all_train_labels,
        all_test_texts,
        all_test_labels
    )
    
    # آموزش مدل
    results = analyzer.train(
        train_dataset, 
        test_dataset,
        output_dir=output_dir,
        epochs=epochs
    )
    
    return analyzer, results

# نمونه استفاده
if __name__ == "__main__":
    # ایجاد آنالایزر چندزبانه
    analyzer = AdvancedMultilingualSentimentAnalyzer()
    
    # بارگذاری مدل و توکنایزر
    analyzer.load_tokenizer()
    analyzer.load_model()
    
    # مثال‌هایی به زبان‌های مختلف
    examples = {
        "fa": [
            "این فیلم واقعاً عالی بود، از دیدنش لذت بردم.",
            "کیفیت محصول متوسط بود، نه خوب نه بد.",
            "خدمات بسیار ضعیف بود و قیمت‌ها خیلی گران."
        ],
        "en": [
            "The movie was amazing, I really enjoyed it.",
            "The product quality was average, neither good nor bad.",
            "The service was terrible and the prices were too high."
        ],
        "fr": [
            "Le film était incroyable, j'ai vraiment apprécié.",
            "La qualité du produit était moyenne, ni bonne ni mauvaise.",
            "Le service était terrible et les prix étaient trop élevés."
        ],
        "de": [
            "Der Film war unglaublich, es hat mir wirklich gefallen.",
            "Die Produktqualität war durchschnittlich, weder gut noch schlecht.",
            "Der Service war schrecklich und die Preise waren zu hoch."
        ],
        "es": [
            "La película fue increíble, realmente la disfruté.",
            "La calidad del producto fue promedio, ni buena ni mala.",
            "El servicio fue terrible y los precios eran demasiado altos."
        ]
    }
    
    # تست مدل با مثال‌های مختلف
    for lang, texts in examples.items():
        print(f"\nتحلیل احساس برای زبان {lang}:")
        
        for text in texts:
            # تشخیص زبان و تحلیل احساس
            result = analyzer.analyze_sentiment_with_language_detection(text)
            
            detected_lang = result['detected_language']
            sentiment = result['sentiment_names'][0]
            score = result['probabilities'][0][result['labels'][0]]
            
            print(f"متن: {text}")
            print(f"زبان تشخیص داده شده: {detected_lang}")
            print(f"احساس: {sentiment} (اطمینان: {score:.2f})")
            print("-" * 50)