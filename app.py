import os, time, json, requests

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "nafisehNik/mt5-persian-summary")

def hf_summarize(text, max_retries=5):
    assert HF_API_KEY, "HF_API_KEY is missing"
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    # T5/mt5 معمولا با این پرامپت بهتر جواب می‌ده
    inp = f"summarize: {text}"

    payload = {
        "inputs": inp,
        "parameters": {
            "max_new_tokens": 180,   # طول خلاصه
            "temperature": 0.2
        },
        "options": {
            "wait_for_model": True   # اگر سرد است، صبر کند
        }
    }

    for attempt in range(1, max_retries + 1):
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        status = r.status_code
        body = None
        try:
            body = r.json()
        except Exception:
            # ممکن است خطای HTML یا متن خام باشد
            body = {"raw": r.text}

        # هندل‌های رایج
        if status == 200:
            # انواع خروجی‌های ممکن
            if isinstance(body, list) and body and isinstance(body[0], dict):
                return body[0].get("summary_text") or body[0].get("generated_text") or json.dumps(body[0], ensure_ascii=False)
            if isinstance(body, dict):
                return body.get("summary_text") or body.get("generated_text") or json.dumps(body, ensure_ascii=False)
            return str(body)

        # Model is loading یا 503 → صبر و تلاش مجدد
        if status in (503, 529) or (isinstance(body, dict) and "estimated_time" in body):
            wait = min(2 ** attempt, 15)
            print(f"[HF] model loading... retry in {wait}s (attempt {attempt}/{max_retries}) | body={body}")
            time.sleep(wait)
            continue

        # Rate limit
        if status == 429:
            wait = min(2 ** attempt, 30)
            print(f"[HF] rate limited. retry in {wait}s | body={body}")
            time.sleep(wait)
            continue

        # توکن نامعتبر/کمبود دسترسی
        if status in (401, 403):
            raise RuntimeError(f"HF auth error ({status}). Check HF_API_KEY and permissions. body={body}")

        # سایر خطاها
        raise RuntimeError(f"HF error {status}: {body}")

    # اگر همه تلاش‌ها ناموفق بود:
    raise RuntimeError(f"HF failed after {max_retries} retries.")
