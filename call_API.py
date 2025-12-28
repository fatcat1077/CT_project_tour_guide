import os, requests

API_KEY = os.getenv("API_KEY", "")
BASE_URL = "https://api-gateway.netdb.csie.ncku.edu.tw"
MODEL = "gemma3:4b"  # 從 /api/tags 找到的 model name

r = requests.post(
    f"{BASE_URL}/api/chat",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "test message"}],
        "stream": False,
    },
    timeout=60,
)
print(r.status_code)
print(r.text)
