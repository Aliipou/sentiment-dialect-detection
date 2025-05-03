import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BertConfig, BertForSequenceClassification
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import os
from pathlib import Path

# تعریف کلاس دیتاست برای ترنسفورمر
class SentimentDataset(Dataset):
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

# کلاس اصلی مدل تشخیص احساسات
class MultilingualSentimentAnalyzer:
    def __init__(self, model_name=None, num_labels=3):
        """
        راه‌اندازی مدل تشخیص احساسات چندزبانه
        
        پارامترها:
            model_name: نام مدل پیش‌آموزش‌دیده (پیش‌فرض: "HooshvareLab/bert-fa-base-uncased" برای فارسی)
            num_labels: تعداد برچسب‌های احساس (پیش‌فرض: 3 - مثبت، خنثی، منفی)
        """
        if model_name is None:
            # برای زبان فارسی از مدل پارسبرت استفاده می‌کنیم
            self.model_name = "HooshvareLab/bert-fa-base-uncased"
        else:
            self.model_name = model_name
            
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
    def load_tokenizer(self):
        """بارگذاری توکنایزر مناسب برای مدل"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer
        
    def load_model(self):
        """بارگذاری مدل پیش‌آموزش‌دیده"""
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
            
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        test_dataset = SentimentDataset(test_texts, test_labels, self.tokenizer)
        
        return train_dataset, test_dataset
    
    def train(self, train_dataset, test_dataset, output_dir="sentiment_model", epochs=3, batch_size=16):
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
        
        trainer.train()
        
        # ارزیابی نهایی
        eval_result = trainer.evaluate()
        
        # ذخیره مدل نهایی
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return eval_result
    
    def predict(self, texts):
        """
        پیش‌بینی احساس برای متن‌های جدید
        
        پارامترها:
            texts: لیستی از متن‌ها
            
        خروجی:
            دیکشنری شامل برچسب‌های پیش‌بینی‌شده و احتمالات
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("مدل یا توکنایزر بارگذاری نشده است")
            
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
        
        result = {
            'labels': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy()
        }
        
        return result
    
    def load_trained_model(self, model_path):
        """
        بارگذاری مدل آموزش‌دیده
        
        پارامترها:
            model_path: مسیر مدل ذخیره‌شده
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)

# نمونه استفاده
if __name__ == "__main__":
    # بارگذاری داده‌ها
    train_persian = pd.read_csv("train_persian.csv")
    test_persian = pd.read_csv("test_persian.csv")
    
    # ایجاد و آموزش مدل
    analyzer = MultilingualSentimentAnalyzer()
    tokenizer = analyzer.load_tokenizer()
    model = analyzer.load_model()
    
    train_dataset, test_dataset = analyzer.create_datasets(
        train_persian['text'].values,
        train_persian['sentiment'].values,
        test_persian['text'].values,
        test_persian['sentiment'].values
    )
    
    # آموزش مدل
    results = analyzer.train(train_dataset, test_dataset)
    print(f"نتایج ارزیابی: {results}")
    
    # پیش‌بینی نمونه
    sample_texts = [
        "این فیلم عالی بود، واقعاً از دیدنش لذت بردم.",
        "کیفیت محصول متوسط بود، نه خوب نه بد.",
        "خدمات بسیار ضعیف بود و قیمت‌ها هم گران."
    ]
    
    predictions = analyzer.predict(sample_texts)
    
    # نمایش نتایج
    sentiment_names = {0: "منفی", 1: "خنثی", 2: "مثبت"}
    for i, text in enumerate(sample_texts):
        label = predictions['labels'][i]
        prob = predictions['probabilities'][i][label]
        print(f"متن: {text}")
        print(f"احساس: {sentiment_names[label]} (اطمینان: {prob:.2f})")
        print("-" * 50)