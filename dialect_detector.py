import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from hazm import Normalizer
import os
import logging

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersianDialectDetector:
    def __init__(self, model_path="models/dialect_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.normalizer = Normalizer()
        
        # لیست لهجه‌های پشتیبانی شده
        self.supported_dialects = ['تهرانی', 'اصفهانی', 'شیرازی', 'مشهدی', 'سایر']
        
        # کلمات کلیدی برای هر لهجه (برای روش قاعده‌محور)
        self.dialect_keywords = {
            'تهرانی': ['داداش', 'مادر', 'دمت گرم', 'ایول', 'جون', 'مشتی', 'بابا'],
            'اصفهانی': ['زِدی', 'خِدا', 'بَرِکَت', 'چِقَد', 'دِس', 'شُد', 'راس'],
            'شیرازی': ['پَ', 'مَنُم', 'اُنجا'],
            'مشهدی': ['آغو', 'بچه‌های مشد'],
            'سایر': ['خواهشمندم', 'اینجانب', 'فرمایید', 'مذکور']
        }

    def load_data(self, file_path):
        """بارگذاری داده‌ها از فایل CSV."""
        try:
            df = pd.read_csv(file_path)
            if 'text' not in df.columns or 'dialect' not in df.columns:
                raise ValueError("فایل باید شامل ستون‌های 'text' و 'dialect' باشد")
            
            # فیلتر کردن لهجه‌های نامعتبر
            df = df[df['dialect'].isin(self.supported_dialects)]
            
            logger.info(f"تعداد {len(df)} نمونه بارگذاری شد")
            return df['text'].astype(str), df['dialect']
        except Exception as e:
            logger.error(f"خطا در بارگذاری داده‌ها: {e}")
            raise

    def train_model(self, data_file='data/persian_dialect_data.csv'):
        """آموزش مدل تشخیص لهجه."""
        try:
            # بررسی وجود فایل
            if not os.path.exists(data_file):
                logger.error(f"فایل {data_file} یافت نشد!")
                return False
            
            X, y = self.load_data(data_file)
            
            # نرمال‌سازی متن‌ها
            X = X.apply(lambda text: self.normalizer.normalize(text))
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"تعداد نمونه‌های آموزش: {len(X_train)}")
            logger.info(f"تعداد نمونه‌های تست: {len(X_test)}")

            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),
                ('clf', MultinomialNB())
            ])

            self.model.fit(X_train, y_train)

            # ارزیابی مدل
            y_pred = self.model.predict(X_test)
            logger.info("گزارش ارزیابی مدل:")
            logger.info(classification_report(y_test, y_pred))

            # ذخیره مدل
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logger.info(f"مدل در {self.model_path} ذخیره شد.")
            
            return True
            
        except Exception as e:
            logger.error(f"خطا در آموزش مدل: {e}")
            return False

    def load_model(self):
        """بارگذاری مدل از فایل."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"مدل از {self.model_path} بارگذاری شد.")
                return True
            else:
                logger.warning(f"فایل مدل در {self.model_path} یافت نشد.")
                return False
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل: {e}")
            return False

    def get_rule_based_prediction(self, text):
        """پیش‌بینی لهجه با روش قاعده‌محور."""
        normalized_text = self.normalizer.normalize(text)
        scores = {}
        
        for dialect, keywords in self.dialect_keywords.items():
            score = sum(1 for keyword in keywords if keyword in normalized_text)
            scores[dialect] = score
        
        if max(scores.values()) == 0:
            return {'label': 4, 'dialect_name': 'سایر', 'confidence': 0.5}
        
        predicted_dialect = max(scores, key=scores.get)
        dialect_index = self.supported_dialects.index(predicted_dialect)
        confidence = scores[predicted_dialect] / sum(scores.values()) if sum(scores.values()) > 0 else 0
        
        return {
            'label': dialect_index,
            'dialect_name': predicted_dialect,
            'confidence': confidence
        }

    def predict_dialect(self, text):
        """تشخیص لهجه متن ورودی."""
        normalized_text = self.normalizer.normalize(text)
        
        # اگر مدل آموزش دیده موجود است
        if self.model is not None:
            predicted = self.model.predict([normalized_text])[0]
            probabilities = self.model.predict_proba([normalized_text])[0]
            label_index = self.supported_dialects.index(predicted)
            
            return {
                'labels': [label_index],
                'dialect_names': [predicted],
                'probabilities': [probabilities]
            }
        
        # در غیر این صورت از روش قاعده‌محور استفاده کن
        rule_based_result = self.get_rule_based_prediction(text)
        return {
            'labels': [rule_based_result['label']],
            'dialect_names': [rule_based_result['dialect_name']],
            'probabilities': [[rule_based_result['confidence']]]
        }

    def load_tokenizer(self):
        """این متد فقط برای سازگاری با API اضافه شده است."""
        return None

    def load_trained_model(self, model_path):
        """بارگذاری مدل آموزش‌دیده."""
        self.model_path = model_path
        return self.load_model()

if __name__ == '__main__':
    detector = PersianDialectDetector()
    
    # اگر مدل آموزش داده نشده است، آن را آموزش دهید
    if not detector.load_model():
        logger.info("مدل یافت نشد. شروع آموزش مدل جدید...")
        if detector.train_model('data/persian_dialect_data.csv'):
            logger.info("آموزش مدل با موفقیت انجام شد.")
        else:
            logger.error("آموزش مدل با شکست مواجه شد.")

    # نمونه پیش‌بینی
    test_examples = [
        "سلام داداش، چطوری؟ دیشب کجا بودی؟",
        "زِدی به بیابون و همه چی رو خراب کِردی؟",
        "لطفاً این موضوع را بررسی کنید."
    ]
    
    for text in test_examples:
        result = detector.predict_dialect(text)
        logger.info(f"متن: {text}")
        logger.info(f"لهجه پیش‌بینی شده: {result['dialect_names'][0]}")
        logger.info("---")