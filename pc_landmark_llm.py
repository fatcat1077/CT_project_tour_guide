import os
import json
import argparse
import requests
from ultralytics import YOLO

# ====== 你自己的景點資料庫（先用硬資訊，之後可再加更多欄位）======
LANDMARK_DB = {
    "fire": {
        "zh_name": "成大光復操場－聖火台",
        "campus": "光復校區",
        "where": "你目前看到的是光復校區的光復操場一帶，聖火台位於操場儀式/司令台區附近。",
        "meaning": "聖火台通常用於大型運動會的點燃儀式，象徵運動精神與傳承，也常是校園大型活動的記憶點。",
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


def extract_llm_text(resp: requests.Response) -> str:
    """
    盡量相容不同 /api/chat 回傳格式：
    - OpenAI-like: {"choices":[{"message":{"content":"..."}}]}
    - OpenWebUI-like: {"message":{"content":"..."}}
    - 其他：直接回 text
    """
    try:
        data = resp.json()
    except Exception:
        return resp.text

    # OpenWebUI style
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
            return data["message"]["content"]

        # OpenAI style
        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
            ch0 = data["choices"][0]
            if isinstance(ch0, dict):
                msg = ch0.get("message", {})
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]

        # 其他常見欄位
        for k in ["content", "response", "text", "answer"]:
            if k in data and isinstance(data[k], str):
                return data[k]

    return resp.text


def call_llm(base_url: str, api_key: str, model: str, messages, timeout: int = 60) -> str:
    r = requests.post(
        f"{base_url}/api/chat",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "stream": False,
        },
        timeout=timeout,
    )

    # 讓錯誤更好 debug
    if not r.ok:
        raise RuntimeError(f"LLM API error {r.status_code}: {r.text}")

    return extract_llm_text(r)


def predict_candidates(yolo_model: YOLO, image_path: str, topk: int = 5):
    res = yolo_model.predict(image_path)
    probs = res[0].probs
    names = res[0].names

    topk_idx = probs.top5[:topk]
    topk_conf = probs.top5conf[:topk]
    candidates = [(names[i], float(c)) for i, c in zip(topk_idx, topk_conf)]
    return candidates


def build_messages(mode: str, pred_class: str, pred_conf: float, info: dict, candidates, user_hint: str = ""):
    # 你可以在這裡固定輸出格式（讓 Quest 顯示更漂亮）
    # 也可以要求 LLM 產 JSON，這裡先讓它輸出可直接顯示的文字。
    sys = (
        "你是成大校園導覽助理。請用繁體中文回答。"
        "不得捏造不存在的史實；若資訊不足請明確說明不確定並給出重拍建議。"
        "回答要適合在 VR 內顯示：段落清楚、不要太長。"
    )

    user = {
        "mode": mode,
        "prediction": {
            "class": pred_class,
            "confidence": round(pred_conf, 4),
            "topk": [(c, round(p, 4)) for c, p in candidates]
        },
        "landmark_info": {
            "name": info["zh_name"],
            "campus": info["campus"],
            "where": info["where"],
            "meaning": info["meaning"],
            "keywords": info.get("keywords", [])
        },
        "user_hint": user_hint
    }

    instruction = (
        "請依 mode 輸出：\n"
        "- where：只回答『這裡是哪裡』\n"
        "- meaning：只回答『這個景點代表什麼意義』\n"
        "- both：分成兩段，先哪裡、再意義\n"
        "最後加上一句『你還想問：附近還有什麼？/怎麼走？』之類的延伸問句。"
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"{instruction}\n\n以下是辨識與景點資料（JSON）：\n{json.dumps(user, ensure_ascii=False, indent=2)}"}
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", default="both", choices=["where", "meaning", "both"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min_conf", type=float, default=0.60)
    parser.add_argument("--yolo_model", default="best.pt")

    # LLM 相關
    parser.add_argument("--base_url", default="https://api-gateway.netdb.csie.ncku.edu.tw")
    parser.add_argument("--llm_model", required=True, help="從 /api/tags 找到的 model name")
    #parser.add_argument("--api_key_env", default="f9f771579808bf8d2c8cb2cf5216d9338be3fbc02fcababaa827639205a06853", help="API key 環境變數名稱")
    parser.add_argument("--user_hint", default="", help="可選：例如『成大校園內』『台南』等提示")

    args = parser.parse_args()

    API_KEY = os.getenv("API_KEY", "")
    if not api_key:
        raise RuntimeError(f"找不到環境變數 {args.api_key_env}，請先設定 API_KEY。")

    yolo = YOLO(args.yolo_model)
    candidates = predict_candidates(yolo, args.image, topk=args.topk)
    print("Top-k candidates:", candidates)

    pred_class, pred_conf = candidates[0]

    # 信心不足：直接回覆「不確定」版本（避免 LLM 胡說）
    if pred_conf < args.min_conf or pred_class not in LANDMARK_DB:
        payload = {
            "status": "unsure",
            "pred_class": pred_class,
            "confidence": pred_conf,
            "topk": candidates,
            "answer": "我不太確定這是哪個景點。你可以更靠近招牌/特徵、換角度或提高亮度再拍一次。",
        }
        print("\n--- JSON payload ---")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    info = LANDMARK_DB[pred_class]

    # 串 LLM
    messages = build_messages(args.mode, pred_class, pred_conf, info, candidates, user_hint=args.user_hint)
    llm_text = call_llm(args.base_url, api_key, args.llm_model, messages)

    payload = {
        "status": "ok",
        "pred_class": pred_class,
        "confidence": pred_conf,
        "place_name": info["zh_name"],
        "campus": info["campus"],
        "mode": args.mode,
        "topk": candidates,
        "answer": llm_text,
        "raw_where": info["where"],
        "raw_meaning": info["meaning"],
    }

    print("\n--- LLM Answer ---")
    print(llm_text)

    print("\n--- JSON payload (for Quest) ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
