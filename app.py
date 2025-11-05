import os, re, time, json, sqlite3, requests
from flask import Flask, request, jsonify, g

# --------- ENV ---------
BOT_TOKEN   = os.getenv("BOT_TOKEN")             # الزامی
HF_API_KEY  = os.getenv("HF_API_KEY")            # الزامی
MODEL_ID    = os.getenv("MODEL_ID", "nafisehNik/mt5-persian-summary")
BOT_USERNAME = (os.getenv("BOT_USERNAME") or "").lower()  # بدون @، اختیاری

DB_PATH = os.getenv("DB_PATH", "messages.db")

# --------- Flask ---------
app = Flask(__name__)

# --------- DB helpers ---------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
        g.db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
          chat_id TEXT,
          message_id INTEGER,
          from_id TEXT,
          date INTEGER,
          text TEXT,
          PRIMARY KEY (chat_id, message_id)
        );
        """)
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def save_message_row(chat_id, message_id, from_id, date_ts, text):
    if not text: return
    db = get_db()
    try:
        db.execute(
            "INSERT OR IGNORE INTO messages(chat_id,message_id,from_id,date,text) VALUES(?,?,?,?,?)",
            (str(chat_id), int(message_id), str(from_id or ""), int(date_ts), text.strip())
        )
        db.commit()
    except Exception as e:
        print("[DB] save error:", e)

def fetch_last_messages(chat_id, n):
    db = get_db()
    rows = db.execute(
        "SELECT text FROM messages WHERE chat_id=? AND text IS NOT NULL ORDER BY message_id DESC LIMIT ?",
        (str(chat_id), int(n))
    ).fetchall()
    return [r["text"] for r in rows]

# --------- Utils ---------
LAST_CMD_RE   = re.compile(r'^/(?:last)(?:@\w+)?\s+(\d+)\b', re.IGNORECASE)
LAST_PLAIN_RE = re.compile(r'^(?:last|آخرین)\s+(\d+)\b', re.IGNORECASE)

def clamp_int(s, lo, hi):
    try: v = int(s)
    except: v = lo
    return max(lo, min(hi, v))

def chunk_by_chars(lines, max_chars=3500):
    chunks, buf, n = [], [], 0
    for t in lines:
        t = (t or "").strip()
        if not t: continue
        if n + len(t) + 1 > max_chars and buf:
            chunks.append("\n".join(buf)); buf, n = [], 0
        buf.append(t); n += len(t) + 1
    if buf: chunks.append("\n".join(buf))
    return chunks

def send_message(chat_id, text, parse_mode=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode: payload["parse_mode"] = parse_mode
    try:
        requests.post(url, json=payload, timeout=30)
    except Exception as e:
        print("[TG] send_message error:", e)

# --------- Hugging Face Summarization (robust) ---------
def hf_summarize(text, max_new_tokens=180, max_retries=5):
    assert HF_API_KEY, "HF_API_KEY missing"
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    inp = f"summarize: {text}"

    payload = {
        "inputs": inp,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2},
        "options": {"wait_for_model": True}
    }

    for attempt in range(1, max_retries + 1):
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        status = r.status_code
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}

        if status == 200:
            if isinstance(body, list) and body and isinstance(body[0], dict):
                return body[0].get("summary_text") or body[0].get("generated_text") or json.dumps(body[0], ensure_ascii=False)
            if isinstance(body, dict):
                return body.get("summary_text") or body.get("generated_text") or json.dumps(body, ensure_ascii=False)
            return str(body)

        # model cold start / over capacity
        if status in (503, 529) or (isinstance(body, dict) and "estimated_time" in body):
            wait = min(2 ** attempt, 15)
            print(f"[HF] loading/capacity. retry in {wait}s ({attempt}/{max_retries}) | {body}")
            time.sleep(wait); continue

        # rate limited
        if status == 429:
            wait = min(2 ** attempt, 30)
            print(f"[HF] rate limited. retry in {wait}s | {body}")
            time.sleep(wait); continue

        if status in (401, 403):
            raise RuntimeError(f"HF auth error {status}: {body}")

        raise RuntimeError(f"HF error {status}: {body}")

    raise RuntimeError("HF failed after retries")

def summarize_last_n(chat_id, n):
    # 1) واکشی آخرین N پیام
    texts = fetch_last_messages(chat_id, n)
    if not texts:
        return "هیچ پیامی ذخیره نشده. بات باید ادمین باشد یا Privacy خاموش شود تا پیام‌ها را ببیند."

    # قدیمی→جدید
    texts = list(reversed([t for t in texts if t.strip()]))

    # 2) چانک و خلاصه‌های میانی
    chunks = chunk_by_chars(texts, max_chars=3500)
    partials = []
    for i, ch in enumerate(chunks, 1):
        try:
            partials.append(hf_summarize(ch, max_new_tokens=180))
        except Exception as e:
            partials.append(f"(خطا در خلاصه‌سازی بخش {i}: {e})")

    # 3) Reduce (خلاصه‌ی نهایی)
    merged = "\n\n".join(partials)
    try:
        final = hf_summarize(
            "خلاصه‌های زیر را به TL;DR فارسی (۳-۵ خط) + نکات کلیدی کوتاه تبدیل کن:\n" + merged,
            max_new_tokens=220
        )
    except Exception:
        final = "\n".join(partials)
    return final

# --------- Routes ---------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "ok": True,
        "bot_token_set": bool(BOT_TOKEN),
        "hf_api_key_set": bool(HF_API_KEY),
        "model": MODEL_ID
    }), 200

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}

    # از آپدیت message و edited_message هر دو لاگ بگیر
    msg = data.get("message") or data.get("edited_message") or {}
    if not msg:
        return "ok", 200

    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    chat_type = chat.get("type")  # private/group/supergroup/channel
    text = (msg.get("text") or "").strip()
    from_user = msg.get("from") or {}
    is_bot_sender = from_user.get("is_bot", False)

    # --- هر پیام دریافتی (متنی) را لاگ کن تا بعداً قابل خلاصه‌سازی باشد ---
    # فقط پیام‌های متنی غیر بات‌ها
    if text and not is_bot_sender and chat_id:
        try:
            save_message_row(
                chat_id=chat_id,
                message_id=msg.get("message_id", 0),
                from_id=from_user.get("id"),
                date_ts=int((msg.get("date") or 0)),
                text=text
            )
        except Exception as e:
            print("[log] save failed:", e)

    if not chat_id or not text:
        return "ok", 200

    # --- منطق پاسخ‌دهی ---
    # PV: /start → راهنما
    if chat_type == "private" and text.startswith("/start"):
        send_message(chat_id,
                     "سلام! برای خلاصه‌سازی بنویس:\n`/last 200`\nیا: `last 200`",
                     "Markdown")
        return "ok", 200

    # گروه/سوپرگروه: فقط وقتی دستور دقیق بود
    if chat_type in ("group", "supergroup"):
        # /last 500  یا  /last@Bot 500
        m = LAST_CMD_RE.match(text)
        if m:
            n = clamp_int(m.group(1), 1, 2000)
            send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ...")
            out = summarize_last_n(chat_id, n)
            send_message(chat_id, out[:4000])
            return "ok", 200

        # last 500 @BotName
        if BOT_USERNAME and (f"@{BOT_USERNAME}" in text.lower()):
            m2 = LAST_PLAIN_RE.search(text)
            if m2:
                n = clamp_int(m2.group(1), 1, 2000)
                send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ...")
                out = summarize_last_n(chat_id, n)
                send_message(chat_id, out[:4000])
                return "ok", 200

        # ریپلای به پیام بات و نوشتن last 100
        reply = msg.get("reply_to_message") or {}
        reply_from = reply.get("from") or {}
        if reply_from.get("is_bot"):
            m3 = LAST_PLAIN_RE.search(text)
            if m3:
                n = clamp_int(m3.group(1), 1, 2000)
                send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ...")
                out = summarize_last_n(chat_id, n)
                send_message(chat_id, out[:4000])
                return "ok", 200

        # غیر از موارد بالا: سکوت کامل در گروه
        return "ok", 200

    # PV: هم /last و هم last
    m = LAST_CMD_RE.match(text) or LAST_PLAIN_RE.match(text)
    if m:
        n = clamp_int(m.group(1), 1, 2000)
        send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ...")
        out = summarize_last_n(chat_id, n)
        send_message(chat_id, out[:4000])
        return "ok", 200

    # PV: پیام راهنما
    if chat_type == "private":
        send_message(chat_id, "برای خلاصه‌سازی بنویس: `/last 200`", "Markdown")
    return "ok", 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
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
