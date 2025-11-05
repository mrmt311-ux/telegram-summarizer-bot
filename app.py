import os
import requests
from flask import Flask, request

BOT_TOKEN = os.environ["BOT_TOKEN"]
HF_API_KEY = os.environ["HF_API_KEY"]
MODEL_ID = "nafisehNik/mt5-persian-summary"

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Bot is running", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    if "message" not in data:
        return "ok", 200

    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")

    if text.startswith("/start"):
        send_message(chat_id, "سلام! من خلاصه‌ساز پیام‌هام. بنویس: `last 10`")
        return "ok", 200

    if text.lower().startswith("last"):
        try:
            n = int(text.split()[1])
        except:
            n = 5
        send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ... (نسخه تست)")
        summary = hf_summarize(f"{n} پیام متنی فرضی برای تست خلاصه‌سازی.")
        send_message(chat_id, summary)
    else:
        send_message(chat_id, "برای تست بنویس: last 5")
    return "ok", 200

def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})

def hf_summarize(text):
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        data = r.json()
        return data[0].get("summary_text") or data[0].get("generated_text") or str(data)
    except Exception as e:
        return f"⚠️ خطا در خلاصه‌سازی: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
