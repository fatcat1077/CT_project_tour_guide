import os
import json
import base64
import time
import re
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
LLM_MODEL = "gemma3:4b"   # hardcode: gpt-oss:120b / gpt-oss:20b / gemma3:4b
YOLO_MODEL_PATH = "best.pt"

TOPK = int(os.getenv("TOPK", "5"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))  # 秒
LLM_RETRY = int(os.getenv("LLM_RETRY", "1"))

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Landmark DB（你可自行擴充）
# =========================
LANDMARK_DB = {
    "sacred_water_machine": {
        "zh_name": "demo用飲水機",
        "campus": "成功校區",
        "where": "你目前看到的是成功校區的資訊系館中的某一台飲水機，這台飲水機既沒有甚麼特別之處，也沒有學生共同回憶。",
        "meaning": "代表學生會在這裡裝水，當然學生也可能在其他飲水機裝水，資訊系可不只有一台飲水機。",
        "keywords": ["資訊系館", "飲水機", "機器", "飲用水", "好喝"]
    },
    "eternal_light": {
        "zh_name": "成大成功校區－永恆之光",
        "campus": "成功校區",
        "where": "你目前所在的位置是成功校區的核心區域一帶，永恆之光設置於校園重要軸線與開放空間交會處，周圍常有學生停留、行走與聚集。",
        "meaning": "永恆之光象徵知識、理想與精神的持續燃燒，代表成功大學對學術追求與公共責任的長久承諾。它不僅是一座校園地標，也承載著世代成大人共同的記憶與價值。",
        "keywords": ["成功校區", "永恆之光", "校園精神", "象徵", "地標"]
    },
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
    mode: str = Field(..., description="where | meaning | both | ask")
    image_b64: str = Field(..., description="base64(jpeg bytes)")
    user_hint: Optional[str] = Field(default="", description="optional hint like '成大校園內'")
    question: Optional[str] = Field(default="", description="when mode=ask, put user's question here")


class AnalyzeResponse(BaseModel):
    request_id: str
    status: str  # ok | unsure | error
    answer: str
    followups: Optional[List[str]] = None  # 兩個追問問題
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
    """兼容 OpenAI-style / Ollama-style / custom gateway"""
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


def _extract_json_object(text: str) -> Optional[dict]:
    """從 LLM 輸出中抓到第一個 JSON object（允許前後有多餘文字）"""
    if not text:
        return None
    s = text.strip()

    # 直接就是 JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 找第一個 { 到最後一個 }
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        snippet = s[l:r + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def parse_llm_answer_followups(text: str) -> Tuple[str, List[str]]:
    """
    寬鬆解析：
    1) 先嘗試抓 JSON: {"answer": "...", "followups": ["..",".."]}
    2) 再找 FOLLOWUP1/2
    3) 再從結尾找兩個像問題的行
    """
    if not text:
        return "", []

    s = text.strip()

    # 1) JSON
    obj = _extract_json_object(s)
    if isinstance(obj, dict):
        ans = (obj.get("answer") or "").strip()
        fu = obj.get("followups") or []
        followups = [q.strip() for q in fu if isinstance(q, str) and q.strip()]
        return ans, followups[:2]

    # 2) FOLLOWUP1/2
    m1 = re.search(r"FOLLOWUP1\s*:\s*(.+)", s)
    m2 = re.search(r"FOLLOWUP2\s*:\s*(.+)", s)
    if m1 and m2:
        f1 = m1.group(1).strip()
        f2 = m2.group(1).strip()
        ans = re.sub(r"FOLLOWUP1\s*:\s*.+", "", s).strip()
        ans = re.sub(r"FOLLOWUP2\s*:\s*.+", "", ans).strip()
        return ans, [f1, f2]

    # 3) 最後兩個像問題的行
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    candidate_q = []
    for ln in lines[::-1]:
        if ln.endswith(("？", "?")) or ln.startswith(("1.", "2.", "-", "•")):
            q = re.sub(r"^(?:\d+\.\s*|[-•]\s*)", "", ln).strip()
            if q:
                candidate_q.append(q)
        if len(candidate_q) >= 2:
            break

    if len(candidate_q) >= 2:
        f2, f1 = candidate_q[0], candidate_q[1]
        ans_lines = lines[:-2]
        ans = "\n".join(ans_lines).strip() if ans_lines else s
        return ans, [f1, f2]

    return s, []


def call_llm(messages) -> str:
    if not API_KEY:
        raise RuntimeError("API_KEY not set")

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


def build_messages(mode: str, info: dict, candidates: List[Tuple[str, float]],
                   user_hint: str = "", user_question: str = ""):
    """
    放寬版本：
    - 不強制只能 JSON
    - 仍「推薦」JSON（Quest 解析更穩）
    - 允許導覽口吻合理延伸（但以 base facts 為核心）
    """
    sys = (
        "你是成大校園導覽助理（繁體中文）。"
        "回答以『提供的基礎資訊』為核心，可以用導覽口吻做合理延伸（例如：怎麼看更清楚、下一步可以怎麼走）。"
        "最後請提出兩個很短的追問問題（適合當按鈕文字）。"
    )

    top1 = candidates[0]
    backup = candidates[1] if len(candidates) > 1 else None

    base_facts = {
        "landmark_name": info["zh_name"],
        "campus": info["campus"],
        "where_fact": info["where"],
        "meaning_fact": info["meaning"],
        "keywords": info.get("keywords", []),
    }

    detect = {
        "top1": {"class": top1[0], "conf": round(top1[1], 4)},
        "backup": {"class": backup[0], "conf": round(backup[1], 4)} if backup else None,
    }

    intent = {"mode": mode, "question": user_question, "hint": user_hint}

    instruction = (
        "請依 mode 回答：where/meaning/both/ask。\n"
        "輸出格式你可以二選一：\n"
        "A)（最推薦）用 JSON：\n"
        "{ \"answer\": \"...\", \"followups\": [\"...\", \"...\"] }\n"
        "B) 或一般文字，但最後一定要有兩行：\n"
        "FOLLOWUP1: ...\n"
        "FOLLOWUP2: ...\n"
        "追問要短（約 8~12 字），像按鈕文字。"
    )

    payload = {
        "intent": intent,
        "detection": detect,
        "base_facts": base_facts,
    }

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": instruction + "\n\n資料（JSON）：\n" + json.dumps(payload, ensure_ascii=False)}
    ]


# =========================
# Vision helpers
# =========================
def predict_candidates(image_path: str, topk: int = TOPK) -> List[Tuple[str, float]]:
    res = yolo_model.predict(image_path)
    probs = res[0].probs
    names = res[0].names
    topk_idx = probs.top5[:topk]
    topk_conf = probs.top5conf[:topk]
    return [(names[i], float(c)) for i, c in zip(topk_idx, topk_conf)]


def pick_best_known_class(candidates: List[Tuple[str, float]]) -> Optional[Tuple[str, float]]:
    """挑最有可能且存在於 LANDMARK_DB 的類別（不管 conf 多低）"""
    for cls, conf in candidates:
        if cls in LANDMARK_DB:
            return cls, conf
    return None


def _default_followups(info: dict) -> List[str]:
    name = info.get("zh_name", "這個景點")
    return [
        f"{name}附近還有什麼？",
        "怎麼走到下一個地點？"
    ]


def _fallback_answer(mode: str, info: dict, user_question: str = "") -> str:
    if mode == "where":
        return info["where"]
    if mode == "meaning":
        return info["meaning"]
    if mode == "ask" and user_question:
        return f"我理解你想問：{user_question}\n\n（先給你目前可用的基礎資訊）\n{info['where']}\n{info['meaning']}"
    return f"這裡是哪裡\n{info['where']}\n\n這個景點代表什麼意義\n{info['meaning']}"


# =========================
# Lifespan
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model

    if yolo_model is None:
        yolo_model = YOLO(YOLO_MODEL_PATH)

    print(f"[startup] YOLO_MODEL_PATH={YOLO_MODEL_PATH}")
    print(f"[startup] BASE_URL={BASE_URL}")
    print(f"[startup] LLM_MODEL={LLM_MODEL}")
    print(f"[startup] API_KEY set? {bool(API_KEY)}")
    yield


app = FastAPI(title="PC Landmark Server", version="1.3", lifespan=lifespan)


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {
        "ok": True,
        "yolo_loaded": yolo_model is not None,
        "api_key_set": bool(API_KEY),
        "base_url": BASE_URL,
        "llm_model": LLM_MODEL
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if yolo_model is None:
        return AnalyzeResponse(
            request_id=req.request_id,
            status="error",
            answer="伺服器 YOLO 尚未載入。",
            followups=["再試一次？", "換角度拍？"],
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
            followups=["重新拍一張？", "檢查連線？"],
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
            "question": req.question,
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
            followups=["再試一次？", "換光線拍？"],
            error=str(e)
        )

    # 4) pick best known class (最可能且在 DB 的)
    best = pick_best_known_class(candidates)
    if best is None:
        return AnalyzeResponse(
            request_id=req.request_id,
            status="unsure",
            answer="我目前找不到對應的景點類別（可能不在資料庫裡）。你可以更靠近招牌/特徵、換角度或提高亮度再拍一次。",
            followups=["要不要重拍？", "換角度試試？"],
            pred_class=candidates[0][0],
            confidence=candidates[0][1],
            topk=[[c, p] for c, p in candidates]
        )

    pred_class, pred_conf = best
    info = LANDMARK_DB[pred_class]

    mode = req.mode if req.mode in ("where", "meaning", "both", "ask") else "both"
    user_question = (req.question or "").strip() if mode == "ask" else ""

    # 5) LLM
    try:
        messages = build_messages(
            mode=mode,
            info=info,
            candidates=candidates,
            user_hint=req.user_hint or "",
            user_question=user_question
        )
        llm_text = call_llm(messages)

        answer, followups = parse_llm_answer_followups(llm_text)

        # answer 保底
        if not answer:
            answer = _fallback_answer(mode, info, user_question=user_question)

        # followups 保底（兩題）
        cleaned = [q.strip() for q in followups if isinstance(q, str) and q.strip()]
        cleaned = cleaned[:2]
        if len(cleaned) < 2:
            cleaned = _default_followups(info)[:2]

        return AnalyzeResponse(
            request_id=req.request_id,
            status="ok",
            answer=answer,
            followups=cleaned,
            pred_class=pred_class,
            confidence=pred_conf,
            place_name=info["zh_name"],
            campus=info["campus"],
            topk=[[c, p] for c, p in candidates],
            raw_where=info["where"],
            raw_meaning=info["meaning"]
        )

    except Exception as e:
        fallback = _fallback_answer(mode, info, user_question=user_question)
        return AnalyzeResponse(
            request_id=req.request_id,
            status="error",
            answer=fallback,
            followups=_default_followups(info)[:2],
            pred_class=pred_class,
            confidence=pred_conf,
            place_name=info["zh_name"],
            campus=info["campus"],
            topk=[[c, p] for c, p in candidates],
            raw_where=info["where"],
            raw_meaning=info["meaning"],
            error=f"LLM error: {e}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
