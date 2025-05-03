import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json

# تنظیم عنوان و توضیحات صفحه
st.set_page_config(
    page_title="تشخیص احساسات و لهجه متن",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# آدرس API
API_URL = "http://localhost:8000"

# تنظیم CSS برای پشتیبانی از فارسی
st.markdown("""
<style>
    @font-face {
        font-family: 'Vazir';
        src: url('https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/Vazir-Regular.ttf');
    }
    
    * {
        font-family: 'Vazir', sans-serif !important;
    }
    
    .rtl {
        direction: rtl;
        text-align: right;
    }
    
    .sentiment-positive {
        color: #1E8449;
        font-weight: bold;
    }
    
    .sentiment-neutral {
        color: #7D3C98;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #C0392B;
        font-weight: bold;
    }
    
    .dialect-tehrani {
        color: #2E86C1;
        font-weight: bold;
    }
    
    .dialect-isfahani {
        color: #D35400;
        font-weight: bold;
    }
    
    .dialect-shirazi {
        color: #27AE60;
        font-weight: bold;
    }
    
    .dialect-mashhadi {
        color: #8E44AD;
        font-weight: bold;
    }
    
    .dialect-other {
        color: #7F8C8D;
        font-weight: bold;
    }
    
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        margin-top: 5px;
    }
    
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# تابع برای اتصال به API و دریافت نتایج
def analyze_text(text, api_endpoint, language="fa"):
    try:
        response = requests.post(
            f"{API_URL}/{api_endpoint}",
            json={"text": text, "language": language}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"خطا در ارتباط با API: {response.text}")
            return None
    except Exception as e:
        st.error(f"خطا در ارتباط با API: {str(e)}")
        return None

def analyze_batch(texts, language="fa"):
    try:
        response = requests.post(
            f"{API_URL}/batch",
            json={"texts": texts, "language": language}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"خطا در ارتباط با API: {response.text}")
            return None
    except Exception as e:
        st.error(f"خطا در ارتباط با API: {str(e)}")
        return None

# رسم نمودار اطمینان
def plot_confidence(score, label_type, label):
    colors = {
        "sentiment": {"مثبت": "#1E8449", "خنثی": "#7D3C98", "منفی": "#C0392B"},
        "dialect": {"تهرانی": "#2E86C1", "اصفهانی": "#D35400", "شیرازی": "#27AE60", "مشهدی": "#8E44AD", "سایر": "#7F8C8D"}
    }
    
    color = colors[label_type].get(label, "#7F8C8D")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[score],
        y=["اطمینان"],
        orientation='h',
        marker=dict(color=color),
        text=[f"{score:.2%}"],
        textposition='auto',
        name=label
    ))
    
    fig.update_layout(
        height=100,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        xaxis=dict(range=[0, 1], showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    return fig

# رسم نمودار مقایسه نتایج گروهی
def plot_batch_results(results):
    sentiments = {"مثبت": 0, "خنثی": 0, "منفی": 0}
    dialects = {"تهرانی": 0, "اصفهانی": 0, "شیرازی": 0, "مشهدی": 0, "سایر": 0}
    
    for result in results:
        sentiments[result["sentiment"]] += 1
        if "dialect" in result and result["dialect"] in dialects:
            dialects[result["dialect"]] += 1
    
    # نمودار احساسات
    sentiment_df = pd.DataFrame({
        "احساس": list(sentiments.keys()),
        "تعداد": list(sentiments.values())
    })
    
    sentiment_colors = {"مثبت": "#1E8449", "خنثی": "#7D3C98", "منفی": "#C0392B"}
    sentiment_fig = px.pie(
        sentiment_df, 
        values="تعداد", 
        names="احساس", 
        title="توزیع احساسات",
        color="احساس",
        color_discrete_map=sentiment_colors
    )
    
    # نمودار لهجه‌ها
    dialect_df = pd.DataFrame({
        "لهجه": list(dialects.keys()),
        "تعداد": list(dialects.values())
    })
    
    dialect_colors = {
        "تهرانی": "#2E86C1", 
        "اصفهانی": "#D35400", 
        "شیرازی": "#27AE60", 
        "مشهدی": "#8E44AD", 
        "سایر": "#7F8C8D"
    }
    
    dialect_fig = px.pie(
        dialect_df, 
        values="تعداد", 
        names="لهجه", 
        title="توزیع لهجه‌ها",
        color="لهجه",
        color_discrete_map=dialect_colors
    )
    
    return sentiment_fig, dialect_fig

# سایدبار برای تنظیمات
st.sidebar.title("تنظیمات")

analysis_mode = st.sidebar.selectbox(
    "حالت تحلیل",
    ["تحلیل متن", "تحلیل گروهی", "آپلود فایل"]
)

language_options = {
    "فارسی": "fa",
    "انگلیسی": "en",
    "فرانسوی": "fr",
    "آلمانی": "de",
    "اسپانیایی": "es",
    "ایتالیایی": "it",
    "پرتغالی": "pt",
    "هلندی": "nl",
    "سوئدی": "sv",
    "نروژی": "no",
    "دانمارکی": "da",
    "فنلاندی": "fi",
    "یونانی": "el",
    "روسی": "ru",
    "لهستانی": "pl",
    "چکی": "cs",
    "مجارستانی": "hu",
    "رومانیایی": "ro",
    "ترکی": "tr"
}

selected_language = st.sidebar.selectbox(
    "انتخاب زبان",
    list(language_options.keys())
)

language_code = language_options[selected_language]

# بررسی ارتباط با API
api_status = st.sidebar.empty()
try:
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        api_status.success("اتصال به API برقرار است")
    else:
        api_status.error("خطا در ارتباط با API")
except:
    api_status.error("عدم دسترسی به API")

# اطلاعات راهنما
with st.sidebar.expander("راهنما"):
    st.markdown("""
    <div class="rtl">
        <h4>نحوه استفاده:</h4>
        <ol>
            <li>متن خود را وارد کنید یا فایل متنی آپلود نمایید</li>
            <li>زبان متن را انتخاب کنید</li>
            <li>برای دریافت نتایج دکمه تحلیل را فشار دهید</li>
        </ol>
        <p>این سیستم قادر به تشخیص احساسات (مثبت، منفی، خنثی) در متن‌های فارسی و اروپایی است. همچنین برای متن‌های فارسی، لهجه نیز تشخیص داده می‌شود.</p>
    </div>
    """, unsafe_allow_html=True)

# صفحه اصلی
st.title("سیستم تشخیص احساسات و لهجه متن")

if analysis_mode == "تحلیل متن":
    st.subheader("تحلیل متن تکی")
    
    text_input = st.text_area(
        "متن خود را وارد کنید:",
        height=150,
        max_chars=1000,
        help="حداکثر 1000 کاراکتر مجاز است"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analyze_button = st.button("تحلیل متن")
    
    if analyze_button and text_input:
        with st.spinner("در حال تحلیل متن..."):
            # دریافت نتایج تحلیل
            result = analyze_text(text_input, "analyze", language_code)
            
            if result:
                st.markdown('<div class="result-box rtl">', unsafe_allow_html=True)
                
                # نمایش متن
                st.markdown(f"<h3>متن ورودی:</h3><p>{result['text']}</p>", unsafe_allow_html=True)
                
                # ستون‌بندی برای نمایش نتایج
                col1, col2 = st.columns(2)
                
                # نمایش احساس
                with col1:
                    sentiment_class = f"sentiment-{result['sentiment'].lower()}" if result['sentiment'] in ["مثبت", "خنثی", "منفی"] else ""
                    st.markdown(f"<h3>احساس تشخیص داده شده:</h3><p class='{sentiment_class}'>{result['sentiment']}</p>", unsafe_allow_html=True)
                    st.plotly_chart(plot_confidence(result['sentiment_score'], "sentiment", result['sentiment']))
                
                # نمایش لهجه (فقط برای فارسی)
                with col2:
                    if language_code == "fa" and "dialect" in result:
                        dialect_class = f"dialect-{result['dialect'].lower()}" if result['dialect'] in ["تهرانی", "اصفهانی", "شیرازی", "مشهدی", "سایر"] else ""
                        st.markdown(f"<h3>لهجه تشخیص داده شده:</h3><p class='{dialect_class}'>{result['dialect']}</p>", unsafe_allow_html=True)
                        st.plotly_chart(plot_confidence(result['dialect_score'], "dialect", result['dialect']))
                    else:
                        st.markdown("<h3>لهجه:</h3><p>تشخیص لهجه فقط برای زبان فارسی فعال است</p>", unsafe_allow_html=True)
                
                st.markdown(f"<p>زمان پردازش: {result['processing_time']:.4f} ثانیه</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif analyze_button and not text_input:
        st.warning("لطفاً متنی را برای تحلیل وارد کنید.")

elif analysis_mode == "تحلیل گروهی":
    st.subheader("تحلیل متن گروهی")
    
    sample_text = """نمونه 1
نمونه 2
نمونه 3"""
    
    batch_texts = st.text_area(
        "متن‌های خود را وارد کنید (هر متن در یک خط):",
        value=sample_text,
        height=200,
        help="هر متن را در یک خط جداگانه وارد کنید"
    )
    
    analyze_batch_button = st.button("تحلیل گروهی")
    
    if analyze_batch_button and batch_texts:
        # تبدیل متن‌ها به لیست
        texts_list = [text.strip() for text in batch_texts.split('\n') if text.strip()]
        
        if len(texts_list) > 0:
            with st.spinner(f"در حال تحلیل {len(texts_list)} متن..."):
                # دریافت نتایج تحلیل گروهی
                batch_result = analyze_batch(texts_list, language_code)
                
                if batch_result and "results" in batch_result:
                    # نمایش آمار کلی
                    st.success(f"تعداد {batch_result['total_texts']} متن با موفقیت تحلیل شد. (زمان کل: {batch_result['total_processing_time']:.2f} ثانیه)")
                    
                    # نمایش نمودارها
                    if len(batch_result['results']) > 1:
                        col1, col2 = st.columns(2)
                        
                        sentiment_fig, dialect_fig = plot_batch_results(batch_result['results'])
                        
                        with col1:
                            st.plotly_chart(sentiment_fig)
                        
                        with col2:
                            if language_code == "fa":
                                st.plotly_chart(dialect_fig)
                    
                    # نمایش جدول نتایج
                    result_df = pd.DataFrame(batch_result['results'])
                    
                    # حذف ستون‌های اضافی
                    columns_to_show = ['text', 'sentiment', 'sentiment_score']
                    if language_code == "fa":
                        columns_to_show.extend(['dialect', 'dialect_score'])
                    
                    if all(col in result_df.columns for col in columns_to_show):
                        st.dataframe(result_df[columns_to_show])
                        
                        # امکان دانلود نتایج
                        csv_data = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="دانلود نتایج (CSV)",
                            data=csv_data,
                            file_name="batch_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        json_data = json.dumps(batch_result, ensure_ascii=False).encode('utf-8')
                        st.download_button(
                            label="دانلود نتایج (JSON)",
                            data=json_data,
                            file_name="batch_analysis_results.json",
                            mime="application/json"
                        )
        else:
            st.warning("هیچ متنی برای تحلیل یافت نشد.")
    
    elif analyze_batch_button and not batch_texts:
        st.warning("لطفاً حداقل یک متن برای تحلیل وارد کنید.")

elif analysis_mode == "آپلود فایل":
    st.subheader("تحلیل فایل متنی")
    
    uploaded_file = st.file_uploader("فایل متنی را آپلود کنید:", type=["txt", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                
                # انتخاب ستون متن
                text_column = st.selectbox("ستون متن را انتخاب کنید:", df.columns.tolist())
                
                if st.button("تحلیل فایل") and text_column:
                    texts_list = df[text_column].dropna().tolist()
                    
                    if len(texts_list) > 0:
                        with st.spinner(f"در حال تحلیل {len(texts_list)} متن از فایل..."):
                            # محدود کردن تعداد متن‌ها برای جلوگیری از فشار بیش از حد به API
                            max_texts = 100
                            if len(texts_list) > max_texts:
                                st.warning(f"برای جلوگیری از فشار بیش از حد به API، فقط {max_texts} متن اول تحلیل می‌شود.")
                                texts_list = texts_list[:max_texts]
                            
                            # دریافت نتایج تحلیل گروهی
                            batch_result = analyze_batch(texts_list, language_code)
                            
                            if batch_result and "results" in batch_result:
                                # افزودن نتایج به DataFrame
                                results_df = pd.DataFrame(batch_result['results'])
                                
                                # ترکیب با DataFrame اصلی
                                result_columns = [col for col in results_df.columns if col != 'text']
                                for col in result_columns:
                                    df[col] = None
                                
                                for i, result in enumerate(batch_result['results']):
                                    text = result['text']
                                    for col in result_columns:
                                        if col in result:
                                            df.loc[df[text_column] == text, col] = result[col]
                                
                                st.success(f"تحلیل {len(texts_list)} متن با موفقیت انجام شد.")
                                
                                # نمایش DataFrame با نتایج
                                st.dataframe(df)
                                
                                # امکان دانلود نتایج
                                csv_data = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="دانلود نتایج (CSV)",
                                    data=csv_data,
                                    file_name="file_analysis_results.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("خطا در دریافت نتایج از API")
                    else:
                        st.warning("فایل خالی است.")
        
        except Exception as e:
            st.error(f"خطا در پردازش فایل: {str(e)}")
    
# فوتر
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #ddd;">
    <p>سیستم تشخیص احساسات و لهجه چندزبانه | نسخه 1.0</p>
</div>
""", unsafe_allow_html=True)
",
                                    mime="text/csv"
                                )
                            else:
                                st.error("خطا در دریافت نتایج از API")
                    else:
                        st.warning("ستون انتخاب شده خالی است.")
            
            else:  # فایل txt
                file_content = uploaded_file.read().decode('utf-8')
                
                # تقسیم به خطوط
                lines = [line.strip() for line in file_content.split('\n') if line.strip()]
                
                st.write(f"فایل با {len(lines)} خط متن بارگذاری شد.")
                
                if st.button("تحلیل فایل"):
                    if len(lines) > 0:
                        with st.spinner(f"در حال تحلیل {len(lines)} متن از فایل..."):
                            # محدود کردن تعداد متن‌ها
                            max_texts = 100
                            if len(lines) > max_texts:
                                st.warning(f"برای جلوگیری از فشار بیش از حد به API، فقط {max_texts} متن اول تحلیل می‌شود.")
                                lines = lines[:max_texts]
                            
                            # دریافت نتایج تحلیل گروهی
                            batch_result = analyze_batch(lines, language_code)
                            
                            if batch_result and "results" in batch_result:
                                # نمایش نتایج
                                results_df = pd.DataFrame(batch_result['results'])
                                st.dataframe(results_df)
                                
                                # امکان دانلود نتایج
                                csv_data = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="دانلود نتایج (CSV)",
                                    data=csv_data,
                                    file_name="file_analysis_results.csv