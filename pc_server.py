import os
import json
import base64
import time
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ultralytics import YOLO


# =========================
# Config
# =========================
BASE_URL = os.getenv("BASE_URL", "https://api-gateway.netdb.csie.ncku.edu.tw")
API_KEY = os.getenv("API_KEY", "")
LLM_MODEL = "gemma3:4b"   # gpt-oss:120b   gpt-oss:20b  gemma3:4b 選一個用
YOLO_MODEL_PATH = "best.pt"

TOPK = int(os.getenv("TOPK", "5"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.60"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))  # 秒
LLM_RETRY = int(os.getenv("LLM_RETRY", "1"))

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Landmark DB（你可自行擴充）
# =========================
LANDMARK_DB = {
    "fire": {
        "zh_name": "成大光復操場－聖火台",
        "campus": "光復校區",
        "where": "你目前看到的是光復校區的光復操場一帶，聖火台位於操場儀式／司令台區附近。",
        "meaning": "聖火台通常用於大型運動會的點燃儀式，象徵運動精神與傳承，也是校園大型活動的重要記憶點。",
        "keywords": ["光復操場", "聖火台", "運動會", "典禮", "儀式"]
    },
    "two_man": {
        "zh_name": "雲平大樓前－飛撲雕像",
        "campus": "光復校區",
        "where": "你目前看到的是光復校區雲平大樓前廣場，前方的雕塑是『飛撲』造型的作品。",
        "meaning": "飛撲的動勢很強，常被解讀成勇敢、投入、彼此互動切磋的意象；在校園裡也很像「衝向目標、共同精進」。",
        "keywords": ["雲平大樓", "飛撲", "雕像", "公共藝術", "動勢"]
    },
    "tree": {
        "zh_name": "光復校區－大榕樹（榕園一帶）",
        "campus": "光復校區",
        "where": "你目前看到的是光復校區榕園附近的大榕樹一帶，是成大很具代表性的校園景點。",
        "meaning": "大榕樹像是校園的時間座標：見證許多人與事件，也象徵「生長、庇蔭、記憶與傳承」。",
        "keywords": ["榕園", "大榕樹", "校園記憶", "遮蔭", "象徵"]
    },
    "climb": {
        "zh_name": "光復校區－攀岩場",
        "campus": "光復校區",
        "where": "你目前看到的是成大校內的人工攀岩場，是一個挑戰型運動設施。",
        "meaning": "攀岩場象徵挑戰與自我突破，也常是訓練、體驗活動與社團練習的場所。",
        "keywords": ["攀岩", "挑戰", "訓練", "運動", "突破"]
    }
}


# =========================
# FastAPI schemas
# =========================
class AnalyzeRequest(BaseModel):
    request_id: str = Field(..., description="client side request id")
    mode: str = Field(..., description="where | meaning | both")
    image_b64: str = Field(..., description="base64(jpeg bytes)")
    user_hint: Optional[str] = Field(default="", description="optional hint like '成大校園內'")


class AnalyzeResponse(BaseModel):
    request_id: str
    status: str  # ok | unsure | error
    answer: str
    pred_class: Optional[str] = None
    confidence: Optional[float] = None
    place_name: Optional[str] = None
    campus: Optional[str] = None
    topk: Optional[List[List[object]]] = None  # [[class, conf], ...]
    raw_where: Optional[str] = None
    raw_meaning: Optional[str] = None
    error: Optional[str] = None


# =========================
# Globals (load once)
# =========================
yolo_model: Optional[YOLO] = None


# =========================
# LLM helpers
# =========================
def _extract_llm_text(resp: requests.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text

    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
            return data["message"]["content"]

        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            ch0 = data["choices"][0]
            if isinstance(ch0, dict):
                msg = ch0.get("message", {})
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]

        for k in ["content", "response", "text", "answer"]:
            if k in data and isinstance(data[k], str):
                return data[k]

    return resp.text


def call_llm(messages) -> str:
    if not API_KEY:
        raise RuntimeError("API_KEY not set")
    if not LLM_MODEL:
        raise RuntimeError("LLM_MODEL not set")

    url = f"{BASE_URL}/api/chat"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
    }

    last_err = None
    for _ in range(LLM_RETRY + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            if not r.ok:
                raise RuntimeError(f"LLM API error {r.status_code}: {r.text}")
            return _extract_llm_text(r)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            last_err = e
            time.sleep(0.8)

    raise RuntimeError(f"LLM timeout after retries: {last_err}")


def build_messages(mode: str, pred_class: str, pred_conf: float, info: dict,
                   candidates: List[Tuple[str, float]], user_hint: str = ""):
    sys = (
        "你是成大校園導覽助理，用繁體中文回答。"
        "不得捏造；若資訊不足請明確說不確定並給出重拍建議。"
        "回答適合 VR 顯示：段落清楚、不要太長。"
    )

    top1 = candidates[0]
    backup = candidates[1] if len(candidates) > 1 else None

    user_data = {
        "mode": mode,
        "pred": {"class": top1[0], "conf": round(top1[1], 3)},
        "backup": {"class": backup[0], "conf": round(backup[1], 3)} if backup else None,
        "name": info["zh_name"],
        "campus": info["campus"],
        "where": info["where"],
        "meaning": info["meaning"],
        "hint": user_hint
    }

    instruction = (
        "依 mode 輸出：where 只回答哪裡；meaning 只回答意義；both 分兩段（先哪裡再意義）。"
        "最後加一句延伸提問，例如：『你還想問附近還有什麼／怎麼走嗎？』"
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": instruction + "\n" + json.dumps(user_data, ensure_ascii=False)}
    ]


def predict_candidates(image_path: str, topk: int = TOPK) -> List[Tuple[str, float]]:
    res = yolo_model.predict(image_path)
    probs = res[0].probs
    names = res[0].names
    topk_idx = probs.top5[:topk]
    topk_conf = probs.top5conf[:topk]
    return [(names[i], float(c)) for i, c in zip(topk_idx, topk_conf)]


# =========================
# Lifespan (replace on_event)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model

    # startup
    if yolo_model is None:
        yolo_model = YOLO(YOLO_MODEL_PATH)

    print(f"[startup] YOLO_MODEL_PATH={YOLO_MODEL_PATH}")
    print(f"[startup] BASE_URL={BASE_URL}")
    print(f"[startup] LLM_MODEL={LLM_MODEL if LLM_MODEL else '(not set)'}")
    print(f"[startup] API_KEY set? {bool(API_KEY)}")

    yield

    # shutdown（目前沒有特別要釋放的，保留給之後用）
    # yolo_model = None


app = FastAPI(title="PC Landmark Server", version="1.1", lifespan=lifespan)


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {
        "ok": True,
        "yolo_loaded": yolo_model is not None,
        "llm_model_set": bool(LLM_MODEL),
        "api_key_set": bool(API_KEY),
        "base_url": BASE_URL,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if yolo_model is None:
        return AnalyzeResponse(
            request_id=req.request_id,
            status="error",
            answer="伺服器 YOLO 尚未載入。",
            error="YOLO not loaded"
        )

    # 1) decode image
    try:
        jpg_bytes = base64.b64decode(req.image_b64)
    except Exception as e:
        return AnalyzeResponse(
            request_id=req.request_id,
            status="error",
            answer="收到的圖片格式錯誤（base64 decode 失敗）。",
            error=str(e)
        )

    # 2) save logs
    img_path = LOG_DIR / f"{req.request_id}.jpg"
    meta_path = LOG_DIR / f"{req.request_id}.meta.json"
    try:
        img_path.write_bytes(jpg_bytes)
        meta_path.write_text(json.dumps({
            "request_id": req.request_id,
            "mode": req.mode,
            "user_hint": req.user_hint,
            "bytes": len(jpg_bytes)
        }, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    # 3) YOLO predict
    try:
        candidates = predict_candidates(str(img_path), topk=TOPK)
    except Exception as e:
        return AnalyzeResponse(
            request_id=req.request_id,
            status="error",
            answer="YOLO 推論失敗。",
            error=str(e)
        )

    pred_class, pred_conf = candidates[0]

    # 4) unsure
    if pred_conf < MIN_CONF or pred_class not in LANDMARK_DB:
        return AnalyzeResponse(
            request_id=req.request_id,
            status="unsure",
            answer="我不太確定這是哪個景點。你可以更靠近招牌/特徵、換角度或提高亮度再拍一次。",
            pred_class=pred_class,
            confidence=pred_conf,
            topk=[[c, p] for c, p in candidates]
        )

    info = LANDMARK_DB[pred_class]
    mode = req.mode if req.mode in ("where", "meaning", "both") else "both"

    # 5) LLM
    try:
        messages = build_messages(mode, pred_class, pred_conf, info, candidates, user_hint=req.user_hint or "")
        llm_text = call_llm(messages)
    except Exception as e:
        # fallback：LLM 失敗仍回可用結果
        if mode == "where":
            fallback = info["where"]
        elif mode == "meaning":
            fallback = info["meaning"]
        else:
            fallback = f"這裡是哪裡\n{info['where']}\n\n這個景點代表什麼意義\n{info['meaning']}"

        return AnalyzeResponse(
            request_id=req.request_id,
            status="error",
            answer=fallback,
            pred_class=pred_class,
            confidence=pred_conf,
            place_name=info["zh_name"],
            campus=info["campus"],
            topk=[[c, p] for c, p in candidates],
            raw_where=info["where"],
            raw_meaning=info["meaning"],
            error=f"LLM error: {e}"
        )

    return AnalyzeResponse(
        request_id=req.request_id,
        status="ok",
        answer=llm_text,
        pred_class=pred_class,
        confidence=pred_conf,
        place_name=info["zh_name"],
        campus=info["campus"],
        topk=[[c, p] for c, p in candidates],
        raw_where=info["where"],
        raw_meaning=info["meaning"]
    )


# =========================
# Allow: py pc_server.py
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
