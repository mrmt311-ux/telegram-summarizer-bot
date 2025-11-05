import os, re, requests
from flask import Flask, request, jsonify

BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "nafisehNik/mt5-persian-summary")
BOT_USERNAME = os.getenv("BOT_USERNAME", "").lower()  # مثلا my_summarizer_bot (بدون @)

app = Flask(__name__)

LAST_CMD_RE = re.compile(r'^/(?:last)(?:@\w+)?\s+(\d+)\b', re.IGNORECASE)
LAST_PLAIN_RE = re.compile(r'^(?:last|آخرین)\s+(\d+)\b', re.IGNORECASE)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"ok": True}), 200

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}
    msg = data.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    chat_type = chat.get("type")  # "private" | "group" | "supergroup" | "channel"

    if not chat_id or not text:
        return "ok", 200

    # --- فقط PV: پیام خوش‌آمد ساده بده ---
    if chat_type == "private" and text.startswith("/start"):
        send_message(chat_id, "سلام! برای خلاصه‌سازی بنویس:\n`/last 500`\nیا: `last 500`", "Markdown")
        return "ok", 200

    # --- منطق گروه: فقط وقتی دقیقاً دستور دادی، جواب بده ---
    if chat_type in ("group", "supergroup"):
        # 1) /last 500  یا  /last@BotName 500
        m = LAST_CMD_RE.match(text)
        if m:
            n = clamp_int(m.group(1), 1, 2000)
            return handle_last(chat_id, n)

        # 2) last 500 @BotName  (بدون اسلش اما با منشنِ بات)
        if BOT_USERNAME and (f"@{BOT_USERNAME}" in text.lower()):
            m2 = LAST_PLAIN_RE.search(text)
            if m2:
                n = clamp_int(m2.group(1), 1, 2000)
                return handle_last(chat_id, n)

        # 3) ریپلای به پیامِ خودِ بات با متن مثل "last 100"
        reply = msg.get("reply_to_message") or {}
        reply_from = (reply.get("from") or {})
        if reply_from.get("is_bot"):
            m3 = LAST_PLAIN_RE.search(text)
            if m3:
                n = clamp_int(m3.group(1), 1, 2000)
                return handle_last(chat_id, n)

        # در غیر این صورت در گروه «هیچ جوابی» نده
        return "ok", 200

    # --- PV: هم /last و هم last پشتیبانی شود ---
    m = LAST_CMD_RE.match(text) or LAST_PLAIN_RE.match(text)
    if m:
        n = clamp_int(m.group(1), 1, 2000)
        return handle_last(chat_id, n)

    # فقط PV: پیام راهنما (در گروه هرگز اسپم نکن)
    if chat_type == "private":
        send_message(chat_id, "برای خلاصه‌سازی بنویس: `/last 200`", "Markdown")
    return "ok", 200

def clamp_int(s, lo, hi):
    try:
        v = int(s)
    except:
        v = lo
    return max(lo, min(hi, v))

def handle_last(chat_id, n):
    # اینجا خلاصه‌سازی واقعی‌ت رو صدا بزن (چانک/Map-Reduce و ...)
    send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ...")
    summary = hf_summarize(f"{n} پیام متنی فرضی برای تست خلاصه‌سازی.")
    send_message(chat_id, summary)
    return "ok", 200

def send_message(chat_id, text, parse_mode=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode: payload["parse_mode"] = parse_mode
    requests.post(url, json=payload, timeout=30)

def hf_summarize(text):
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
    payload = {"inputs": f"summarize: {text}", "options": {"wait_for_model": True}}
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    try:
        data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0].get("summary_text") or data[0].get("generated_text") or str(data[0])
        if isinstance(data, dict):
            return data.get("summary_text") or data.get("generated_text") or str(data)
        return str(data)
    except Exception as e:
        return f"⚠️ خطا در خلاصه‌سازی: {e}"
