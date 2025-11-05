import os, re, time, json, sqlite3, requests
from flask import Flask, request, jsonify, g

# ========= ENV =========
BOT_TOKEN    = os.getenv("BOT_TOKEN")                       # الزامی (از BotFather)
HF_API_KEY   = os.getenv("HF_API_KEY")                      # الزامی (از HF Settings > Access Tokens)
MODEL_ID     = os.getenv("MODEL_ID", "nafisehNik/mt5-persian-summary")
BOT_USERNAME = (os.getenv("BOT_USERNAME") or "").lower()    # بدون @ (مثال: telegram_summarizer_bot)
DB_PATH      = os.getenv("DB_PATH", "messages.db")          # مسیر پایگاه‌داده SQLite

# ========= FLASK =========
app = Flask(__name__)

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

def send_message(chat_id, text, parse_mode=None):
    if not BOT_TOKEN: 
        print("[TG] Missing BOT_TOKEN")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        requests.post(url, json=payload, timeout=30)
    except Exception as e:
        print("[TG] send_message error:", e)

# ========= HF SUMMARIZER (Router) =========
# NOTE: api-inference.huggingface.co deprecated → use router.huggingface.co/hf-inference
def hf_summarize(text, max_new_tokens=180, max_retries=5):
    assert HF_API_KEY, "HF_API_KEY missing"
    url = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # مدل‌های T5/mt5 با پرامپت summarize: نتیجه‌ی بهتری می‌دهند
    payload = {
        "inputs": f"summarize: {text}",
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.2,
            "do_sample": False
        },
        "options": {"wait_for_model": True}
    }

    last_raw = None
    for attempt in range(1, max_retries + 1):
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        status = r.status_code
        raw = (r.text or "").strip()
        last_raw = raw

        body = None
        if raw:
            try:
                body = r.json()
            except Exception:
                body = None

        # Success with JSON
        if status == 200 and body is not None:
            if isinstance(body, list) and body and isinstance(body[0], dict):
                return body[0].get("summary_text") or body[0].get("generated_text") or json.dumps(body[0], ensure_ascii=False)
            if isinstance(body, dict):
                return body.get("summary_text") or body.get("generated_text") or json.dumps(body, ensure_ascii=False)
            return str(body)

        # 200 ولی بدنه خالی/غیر JSON → retry
        if status == 200 and not raw:
            wait = min(2 ** attempt, 10)
            print(f"[HF] empty body; retry in {wait}s ({attempt}/{max_retries})")
            time.sleep(wait); continue

        # مدل مشغول/در حال بارگذاری
        if status in (503, 529) or (isinstance(body, dict) and body.get("estimated_time")):
            wait = min(2 ** attempt, 15)
            print(f"[HF] loading/capacity; retry in {wait}s | body={body or raw[:200]}")
            time.sleep(wait); continue

        # ریت‌لیمیت
        if status == 429:
            wait = min(2 ** attempt, 30)
            print(f"[HF] rate limited; retry in {wait}s")
            time.sleep(wait); continue

        # احراز هویت/دسترسی
        if status in (401, 403):
            raise RuntimeError(f"HF auth error {status}: {body or raw[:500]}")

        # سایر خطاها
        raise RuntimeError(f"HF error {status}: {body or raw[:500]}")

    raise RuntimeError(f"HF failed after retries. last_raw={last_raw[:500] if last_raw else 'EMPTY'}")

def summarize_last_n(chat_id, n):
    # 1) واکشی آخرین N پیام از همین چت
    texts = fetch_last_texts(chat_id, n)
    if not texts:
        return "هیچ پیامی ذخیره نشده. ربات باید ادمین باشد یا Group Privacy خاموش شود تا پیام‌ها را ببیند."

    # قدیمی→جدید و حذف خالی‌ها
    texts = list(reversed([t for t in texts if t.strip()]))

    # 2) چانک + خلاصه‌های میانی
    chunks = chunk_by_chars(texts, max_chars=3500)
    partials = []
    for i, ch in enumerate(chunks, 1):
        try:
            partials.append(hf_summarize(ch, max_new_tokens=180))
        except Exception as e:
            partials.append(f"(خطا در خلاصه‌سازی بخش {i}: {e})")

    # 3) Reduce (جمع‌بندی نهایی)
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

    # message یا edited_message
    msg = data.get("message") or data.get("edited_message") or {}
    if not msg:
        return "ok", 200

    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    chat_type = chat.get("type")  # private/group/supergroup/channel
    text = (msg.get("text") or "").strip()
    from_user = msg.get("from") or {}
    is_bot_sender = from_user.get("is_bot", False)

    # هر پیام متنی از غیر‌بات را ذخیره کن
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

    # PV: /start → راهنما
    if chat_type == "private" and text.startswith("/start"):
        send_message(chat_id,
                     "سلام! برای خلاصه‌سازی بنویس:\n`/last 200`\nیا: `last 200`",
                     "Markdown")
        return "ok", 200

    # گروه‌ها: فقط وقتی «واقعاً» دستور داده شد (بدون اسپم)
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

        # ریپلای به پیامِ خودِ بات و تایپ last 100
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

        # در غیر این صورت سکوت
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
