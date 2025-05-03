#   پیاده‌سازی مدل تشخیص لهجه فارسی
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from hazm import Normalizer

class PersianDialectDetector:
    def __init__(self, model_path="dialect_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.normalizer = Normalizer()  # استفاده از هزیم برای نرمال‌سازی متن

    def load_data(self, file_path):
        """بارگیری داده‌ها از فایل CSV."""

        df = pd.read_csv(file_path)
        return df['text'].astype(str), df['dialect']

    def train_model(self, data_file):
        """آموزش مدل تشخیص لهجه."""

        X, y = self.load_data(data_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),  #  می‌توانید پارامترها را تنظیم کنید
            ('clf', MultinomialNB())
        ])

        self.model.fit(X_train, y_train)

        # ارزیابی مدل
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # ذخیره مدل
        joblib.dump(self.model, self.model_path)
        print(f"مدل در {self.model_path} ذخیره شد.")

    def load_model(self):
        """بارگیری مدل از فایل."""

        self.model = joblib.load(self.model_path)
        print(f"مدل از {self.model_path} بارگیری شد.")

    def predict_dialect(self, text):
        """تشخیص لهجه متن ورودی."""
        text = self.normalizer.normalize(text)  # نرمال‌سازی متن ورودی
        if self.model is None:
            self.load_model()
        return self.model.predict([text])[0]

if __name__ == '__main__':
    detector = PersianDialectDetector()
    # اگر مدل آموزش داده نشده است، آن را آموزش دهید
    try:
        detector.load_model()
    except FileNotFoundError:
        detector.train_model('dialect_data.csv')  #  مسیر فایل داده‌های آموزشی خود را وارد کنید

    #  نمونه پیش‌بینی
    text_to_predict = "ای بابا چه خبره؟"
    predicted_dialect = detector.predict_dialect(text_to_predict)
    print(f"متن: {text_to_predict}، لهجه پیش‌بینی شده: {predicted_dialect}")