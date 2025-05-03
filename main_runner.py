#   اسکریپت اصلی برای راه‌اندازی پروژه
import uvicorn
import subprocess
import time
import os

def run_api():
    """راه‌اندازی API با استفاده از uvicorn."""
    print("در حال راه‌اندازی API...")
    uvicorn.run("sentiment_api:app", host="0.0.0.0", port=8000, reload=True)  #  تغییر پورت در صورت نیاز

def run_streamlit_app():
    """راه‌اندازی برنامه Streamlit."""
    print("در حال راه‌اندازی برنامه Streamlit...")
    streamlit_path = "streamlit_app.py"  #  مسیر فایل Streamlit خود را وارد کنید
    subprocess.Popen(["streamlit", "run", streamlit_path])

if __name__ == "__main__":
    print("شروع راه‌اندازی پروژه...")

    #  راه‌اندازی API (در یک thread جداگانه یا فرآیند جداگانه)
    #  اینجا از subprocess استفاده شده است، می‌توانید از threading هم استفاده کنید
    api_process = subprocess.Popen(["python", "-c", "import uvicorn; uvicorn.run('sentiment_api:app', host='0.0.0.0', port=8000, reload=True)"])

    #  کمی صبر کنید تا API راه‌اندازی شود
    time.sleep(5)

    #  راه‌اندازی برنامه Streamlit
    run_streamlit_app()

    print("پروژه با موفقیت راه‌اندازی شد. برای بستن، Ctrl+C را فشار دهید.")

    #  نگه داشتن فرآیند اصلی برای جلوگیری از بسته شدن فوری
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("در حال بستن پروژه...")
        api_process.terminate()  # بستن فرآیند API
        print("پروژه بسته شد.")