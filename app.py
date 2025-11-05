# app.py
import os, re, time, sqlite3, requests, logging, threading
from flask import Flask, request, jsonify, g
from huggingface_hub import InferenceClient

# ==== Logging ====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== ENV ====
BOT_TOKEN    = os.getenv("BOT_TOKEN")
HF_API_KEY   = os.getenv("HF_API_KEY")
MODEL_ID     = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
BOT_USERNAME = (os.getenv("BOT_USERNAME") or "").lower().lstrip("@")
DB_PATH      = os.getenv("DB_PATH", "/tmp/messages.db")
CLEANUP_DAYS = int(os.getenv("CLEANUP_DAYS", "30"))
KEEP_ALIVE_URL = os.getenv("RENDER_EXTERNAL_URL")

# ==== Flask ====
app = Flask(__name__)

# ==== HF Client (آدرس جدید) ====
_HF_CLIENT = None
def get_hf_client():
    global _HF_CLIENT
    if _HF_CLIENT is None:
        assert HF_API_KEY, "HF_API_KEY missing"
        _HF_CLIENT = InferenceClient(
            model=MODEL_ID,
            token=HF_API_KEY,
            timeout=120,
            base_url="https://router.huggingface.co/hf-inference"  # آدرس جدید
        )
    return _HF_CLIENT

def hf_summarize(text, max_new_tokens=180, max_retries=5):
    client = get_hf_client()
    text = text.strip()
    model_lower = MODEL_ID.lower()

    if "t5" in model_lower or "mt5" in model_lower:
        prompt = f"summarize: {text}"
        for attempt in range(1, max_retries + 1):
            try:
                out = client.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    do_sample=False,
                    repetition_penalty=1.1,
                    return_full_text=False,
                )
                return (out or "").strip()
            except Exception as e:
                wait = min(2 ** attempt, 12)
                logger.warning(f"[HF] T5 attempt {attempt} failed: {e}")
                time.sleep(wait)
        raise RuntimeError("T5 model failed")

    else:
        messages = [{"role": "user", "content": f"خلاصه فارسی کن (حداکثر 150 کلمه):\n\n{text}"}]
        for attempt in range(1, max_retries + 1):
            try:
                out = client.chat_completion(
                    messages,
                    max_tokens=max_new_tokens,
                    temperature=0.2,
                    stream=False,
                )
                return out.choices[0].message.content.strip()
            except Exception as e:
                wait = min(2 ** attempt, 12)
                logger.warning(f"[HF] Instruct attempt {attempt} failed: {e}")
                time.sleep(wait)
        raise RuntimeError("Instruct model failed")

# ==== DB (SQLite) ====
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL;")
        g.db.execute("PRAGMA synchronous=NORMAL;")
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
            "INSERT OR REPLACE INTO messages(chat_id,message_id,from_id,date,text) VALUES(?,?,?,?,?)",
            (str(chat_id), int(message_id or 0), str(from_id or ""), int(date_ts or 0), text.strip())
        )
        db.commit()
    except Exception as e:
        logger.error(f"[DB] save error: {e}")

def fetch_last_texts(chat_id, n):
    db = get_db()
    rows = db.execute(
        "SELECT text FROM messages WHERE chat_id=? AND text IS NOT NULL AND text != '' ORDER BY message_id DESC LIMIT ?",
        (str(chat_id), int(n))
    ).fetchall()
    return [r["text"] for r in rows]

def cleanup_old_messages(days=CLEANUP_DAYS):
    if days <= 0: return
    db = get_db()
    cutoff = int(time.time()) - (days * 86400)
    try:
        deleted = db.execute("DELETE FROM messages WHERE date < ?", (cutoff,)).rowcount
        db.commit()
        if deleted:
            logger.info(f"[DB] cleanup: {deleted} old messages deleted")
    except Exception as e:
        logger.error(f"[DB] cleanup error: {e}")

# ==== Utils ====
LAST_CMD_RE   = re.compile(r'^/(?:last)(?:@\w+)?\s+(\d+)\b', re.IGNORECASE)
LAST_PLAIN_RE = re.compile(r'^(?:last|آخرین)\s+(\d+)\b', re.IGNORECASE)
CLEAR_RE      = re.compile(r'^/clear\b', re.IGNORECASE)

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
            chunks.append("\n".join(buf))
            buf, n = [], 0
        buf.append(t)
        n += len(t) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks

def split_message(text, limit=4000):
    parts = []
    s = (text or "").rstrip()
    while len(s) > limit:
        idx = s.rfind("\n", 0, limit)
        if idx == -1: idx = s.rfind(" ", 0, limit)
        if idx == -1: idx = limit
        parts.append(s[:idx])
        s = s[idx:].lstrip("\n ")
    if s:
        parts.append(s)
    return parts

def escape_markdown_v2(text):
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def send_message(chat_id, text, parse_mode=None, reply_to_message_id=None):
    if not BOT_TOKEN:
        logger.error("[TG] Missing BOT_TOKEN")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    parts = split_message(text, 4000)
    for i, part in enumerate(parts):
        if i > 0:
            time.sleep(0.35)
        if parse_mode == "MarkdownV2":
            part = escape_markdown_v2(part)
        payload = {
            "chat_id": chat_id,
            "text": part or " ",
            "parse_mode": parse_mode,
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code != 200:
                logger.warning(f"[TG] send failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.error(f"[TG] send_message error: {e}")

def summarize_last_n(chat_id, n):
    cleanup_old_messages()
    texts = fetch_last_texts(chat_id, n)
    if not texts:
        return "هیچ پیامی ذخیره نشده.\n\nبرای گروه: ربات را **Admin** کنید."

    texts = list(reversed([t for t in texts if t.strip()]))
    chunks = chunk_by_chars(texts, max_chars=3500)

    partials = []
    for i, ch in enumerate(chunks, 1):
        try:
            partial = hf_summarize(ch, max_new_tokens=180)
            partials.append(partial)
        except Exception as e:
            partials.append(f"(خطا در خلاصه‌سازی بخش {i})")

    merged = "\n\n".join(partials)
    try:
        final_prompt = (
            "خلاصه‌های زیر را به یک خلاصه حرفه‌ای تبدیل کن:\n"
            "- TL;DR (۳–۵ خط)\n"
            "- نکات کلیدی\n"
            "- تصمیم‌ها\n\n" + merged
        )
        final = hf_summarize(final_prompt, max_new_tokens=250)
    except Exception:
        final = "\n\n".join(partials)

    return final

# ==== Keep-Alive ===
def keep_alive():
    if not KEEP_ALIVE_URL:
        return
    def ping():
        while True:
            try:
                requests.get(KEEP_ALIVE_URL, timeout=10)
                logger.info("[KEEP-ALIVE] Ping sent")
            except:
                pass
            time.sleep(240)
    thread = threading.Thread(target=ping, daemon=True)
    thread.start()

# ==== Routes ====
@app.route("/", methods=["GET"])
def health():
    try:
        get_db().execute("SELECT 1").fetchone()
        get_hf_client()
        return jsonify({"ok": True, "db": "connected", "hf": "ready"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/webhook", methods=["POST", "GET"])
def webhook():
    if request.method == "GET":
        return "ok", 200  # برای تست دستی

    data = request.get_json(silent=True) or {}
    msg = data.get("message") or data.get("edited_message") or {}
    if not msg:
        return "ok", 200

    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    chat_type = chat.get("type")
    text = (msg.get("text") or msg.get("caption") or "").strip()
    from_user = msg.get("from") or {}
    is_bot_sender = from_user.get("is_bot", False)
    message_id = msg.get("message_id")
    date_ts = int(msg.get("date") or 0)

    if text and not is_bot_sender and chat_id:
        save_message_row(chat_id, message_id, from_user.get("id"), date_ts, text)

    if not chat_id or not text:
        return "ok", 200

    if text.lower().startswith("/ping"):
        send_message(chat_id, "pong", "MarkdownV2")
        return "ok", 200

    if text.lower().startswith("/count"):
        db = get_db()
        c = db.execute("SELECT COUNT(*) AS c FROM messages WHERE chat_id=?", (str(chat_id),)).fetchone()
        total = c["c"] if c else 0
        send_message(chat_id, f"*پیام‌های ذخیره‌شده:* `{total}`", "MarkdownV2")
        return "ok", 200

    if chat_type == "private" and text.startswith("/start"):
        send_message(chat_id, "*ربات خلاصه‌ساز*\n\n`/last 200`", "MarkdownV2")
        return "ok", 200

    if chat_type == "private" and CLEAR_RE.match(text):
        db = get_db()
        db.execute("DELETE FROM messages WHERE chat_id=?", (str(chat_id),))
        db.commit()
        send_message(chat_id, "تاریخچه پاک شد.", "MarkdownV2")
        return "ok", 200

    if chat_type in ("group", "supergroup"):
        reply = msg.get("reply_to_message") or {}
        reply_from = reply.get("from") or {}
        is_reply_to_bot = reply_from.get("is_bot", False)

        m = LAST_CMD_RE.match(text)
        if m:
            n = clamp_int(m.group(1), 1, 2000)
            send_message(chat_id, f"در حال خلاصه‌سازی *{n}* پیام...", "MarkdownV2")
            summary = summarize_last_n(chat_id, n)
            send_message(chat_id, summary, "MarkdownV2")
            return "ok", 200

        if BOT_USERNAME and f"@{BOT_USERNAME}" in text.lower():
            m2 = LAST_PLAIN_RE.search(text)
            if m2:
                n = clamp_int(m2.group(1), 1, 2000)
                send_message(chat_id, f"در حال خلاصه‌سازی *{n}* پیام...", "MarkdownV2")
                summary = summarize_last_n(chat_id, n)
                send_message(chat_id, summary, "MarkdownV2")
                return "ok", 200

        if is_reply_to_bot:
            m3 = LAST_PLAIN_RE.search(text)
            if m3:
                n = clamp_int(m3.group(1), 1, 2000)
                send_message(chat_id, f"در حال خلاصه‌سازی *{n}* پیام...", "MarkdownV2")
                summary = summarize_last_n(chat_id, n)
                send_message(chat_id, summary, "MarkdownV2", reply_to_message_id=message_id)
                return "ok", 200

        return "ok", 200

    m = LAST_CMD_RE.match(text) or LAST_PLAIN_RE.match(text)
    if m:
        n = clamp_int(m.group(1), 1, 2000)
        send_message(chat_id, f"در حال خلاصه‌سازی *{n}* پیام...", "MarkdownV2")
        summary = summarize_last_n(chat_id, n)
        send_message(chat_id, summary, "MarkdownV2")
        return "ok", 200

    if chat_type == "private":
        send_message(chat_id, "بنویس: `/last 200`", "MarkdownV2")

    return "ok", 200

# ==== Run ====
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    keep_alive()
    app.run(host="0.0.0.0", port=port, debug=False)
