import os, re, time, json, sqlite3, requests
from flask import Flask, request, jsonify, g
from huggingface_hub import InferenceClient

# ========= ENV =========
BOT_TOKEN    = os.getenv("BOT_TOKEN")                       # الزامی (BotFather)
HF_API_KEY   = os.getenv("HF_API_KEY")                      # الزامی (HF > Settings > Access Tokens)
MODEL_ID     = os.getenv("MODEL_ID", "nafisehNik/mt5-persian-summary")
BOT_USERNAME = (os.getenv("BOT_USERNAME") or "").lower()    # بدون @ (مثال: telegram_summarizer_bot)
DB_PATH      = os.getenv("DB_PATH", "messages.db")

# ========= FLASK =========
app = Flask(__name__)

# ========= HF CLIENT (Router) =========
# از روتر جدید Hugging Face استفاده می‌کنیم
_HF_CLIENT = None
def get_hf_client():
    global _HF_CLIENT
    if _HF_CLIENT is None:
        assert HF_API_KEY, "HF_API_KEY missing"
        _HF_CLIENT = InferenceClient(
            model=MODEL_ID,
            token=HF_API_KEY,
            base_url="https://router.huggingface.co",
            timeout=90
        )
    return _HF_CLIENT

def hf_summarize(text, max_new_tokens=180, max_retries=5):
    """
    مدل‌های MT5/T5 باید با text2text صدا زده شوند.
    prefix 'summarize:' هم کیفیت بهتری می‌دهد.
    """
    client = get_hf_client()
    prompt = f"summarize: {text}"
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            out = client.text2text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False,
            )
            return (out or "").strip()
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 15)
            print(f"[HF] error attempt {attempt}/{max_retries}: {e} | retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"HF failed after retries: {last_err}")

# ========= DB =========
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
def close_db(_):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def save_message_row(chat_id, message_id, from_id, date_ts, text):
    if not text: return
    db = get_db()
    try:
        db.execute(
            "INSERT OR IGNORE INTO messages(chat_id,message_id,from_id,date,text) VALUES(?,?,?,?,?)",
            (str(chat_id), int(message_id or 0), str(from_id or ""), int(date_ts or 0), text.strip())
        )
        db.commit()
    except Exception as e:
        print("[DB] save error:", e)

def fetch_last_texts(chat_id, n):
    db = get_db()
    rows = db.execute(
        "SELECT text FROM messages WHERE chat_id=? AND text IS NOT NULL ORDER BY message_id DESC LIMIT ?",
        (str(chat_id), int(n))
    ).fetchall()
    return [r["text"] for r in rows]

# ========= UTILS =========
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

def split_message(text, limit=4000):
    """تقسیم متن بلند به پیام‌های <= 4000 کاراکتری برای تلگرام"""
    parts = []
    s = text or ""
    while len(s) > limit:
        idx = s.rfind("\n", 0, limit)
        if idx == -1: idx = limit
        parts.append(s[:idx])
        s = s[idx:].lstrip("\n")
    parts.append(s)
    return parts

def send_message(chat_id, text, parse_mode=None):
    if not BOT_TOKEN:
        print("[TG] Missing BOT_TOKEN"); return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for part in split_message(text, 4000):
        payload = {"chat_id": chat_id, "text": part}
        if parse_mode: payload["parse_mode"] = parse_mode
        try:
            requests.post(url, json=payload, timeout=30)
        except Exception as e:
            print("[TG] send_message error:", e)

def summarize_last_n(chat_id, n):
    texts = fetch_last_texts(chat_id, n)
    if not texts:
        return "هیچ پیامی ذخیره نشده. ربات باید ادمین باشد یا Group Privacy خاموش شود تا پیام‌ها را ببیند."

    texts = list(reversed([t for t in texts if t.strip()]))  # قدیمی→جدید
    chunks = chunk_by_chars(texts, max_chars=3500)

    partials = []
    for i, ch in enumerate(chunks, 1):
        try:
            partials.append(hf_summarize(ch, max_new_tokens=180))
        except Exception as e:
            partials.append(f"(خطا در خلاصه‌سازی بخش {i}: {e})")

    merged = "\n\n".join(partials)
    try:
        final = hf_summarize(
            "این خلاصه‌های میانی را به یک خروجی تمیز تبدیل کن:\n"
            "- TL;DR (۳–۵ خط)\n- نکات کلیدی فهرست‌وار\n- تصمیم‌ها/اکشن‌ها (اگر بود)\n\n" + merged,
            max_new_tokens=220
        )
    except Exception:
        final = "\n".join(partials)
    return final

# ========= ROUTES =========
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "bot_token_set": bool(BOT_TOKEN),
        "hf_api_key_set": bool(HF_API_KEY),
        "model": MODEL_ID
    }), 200

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}

    msg = data.get("message") or data.get("edited_message") or {}
    if not msg:
        return "ok", 200

    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    chat_type = chat.get("type")  # private/group/supergroup/channel
    text = (msg.get("text") or "").strip()
    from_user = msg.get("from") or {}
    is_bot_sender = from_user.get("is_bot", False)

    # ذخیره همه پیام‌های متنی (غیر بات‌ها)
    if text and not is_bot_sender and chat_id:
        try:
            save_message_row(
                chat_id=chat_id,
                message_id=msg.get("message_id", 0),
                from_id=from_user.get("id"),
                date_ts=int(msg.get("date") or 0),
                text=text
            )
        except Exception as e:
            print("[DB] save failed:", e)

    if not chat_id or not text:
        return "ok", 200

    # دستورات دیباگ
    if text.lower().startswith("/ping"):
        send_message(chat_id, "pong ✅")
        return "ok", 200

    if text.lower().startswith("/count"):
        db = get_db()
        c = db.execute("SELECT COUNT(*) AS c FROM messages WHERE chat_id=?", (str(chat_id),)).fetchone()
        total = c["c"] if c else 0
        send_message(chat_id, f"تعداد پیام‌های ذخیره‌شده برای این چت: {total}")
        return "ok", 200

    # PV: /start → راهنما
    if chat_type == "private" and text.startswith("/start"):
        send_message(chat_id,
                     "سلام! برای خلاصه‌سازی بنویس:\n`/last 200`\nیا: `last 200`",
                     "Markdown")
        return "ok", 200

    # گروه‌ها: فقط در حالت‌های مجاز پاسخ بده (بدون اسپم)
    if chat_type in ("group", "supergroup"):
        # 1) /last 500  یا  /last@Bot 500
        m = LAST_CMD_RE.match(text)
        if m:
            n = clamp_int(m.group(1), 1, 2000)
            send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ...")
            out = summarize_last_n(chat_id, n)
            send_message(chat_id, out)
            return "ok", 200

        # 2) last 500 @BotName
        if BOT_USERNAME and (f"@{BOT_USERNAME}" in text.lower()):
            m2 = LAST_PLAIN_RE.search(text)
            if m2:
                n = clamp_int(m2.group(1), 1, 2000)
                send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ...")
                out = summarize_last_n(chat_id, n)
                send_message(chat_id, out)
                return "ok", 200

        # 3) ریپلای به پیام خودِ بات و تایپ last 100
        reply = msg.get("reply_to_message") or {}
        reply_from = reply.get("from") or {}
        if reply_from.get("is_bot"):
            m3 = LAST_PLAIN_RE.search(text)
            if m3:
                n = clamp_int(m3.group(1), 1, 2000)
                send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ...")
                out = summarize_last_n(chat_id, n)
                send_message(chat_id, out)
                return "ok", 200

        # در غیر این صورت سکوت
        return "ok", 200

    # PV: هم /last و هم last
    m = LAST_CMD_RE.match(text) or LAST_PLAIN_RE.match(text)
    if m:
        n = clamp_int(m.group(1), 1, 2000)
        send_message(chat_id, f"در حال خلاصه‌سازی آخرین {n} پیام ...")
        out = summarize_last_n(chat_id, n)
        send_message(chat_id, out)
        return "ok", 200

    # PV: پیام راهنما
    if chat_type == "private":
        send_message(chat_id, "برای خلاصه‌سازی بنویس: `/last 200`", "Markdown")
    return "ok", 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
