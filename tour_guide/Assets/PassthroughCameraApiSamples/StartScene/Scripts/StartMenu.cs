// Copyright (c) Meta Platforms, Inc. and affiliates.
// Modified for: 2-button landmark QA + send frame to PC server + show answer (BIG BOX)

using System;
using System.Collections;
using System.Text;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace PassthroughCameraSamples.StartScene
{
    [MetaCodeSample("PassthroughCameraApiSamples-StartScene")]
    public class StartMenu : MonoBehaviour
    {
        [Header("PC Server (USB 建議用 adb reverse，Quest 端打 127.0.0.1)")]
        [Tooltip("例：http://127.0.0.1:8000/analyze")]
        public string AnalyzeUrl = "http://127.0.0.1:8000/analyze";

        [Tooltip("JPEG 壓縮品質 1~100，越大越清晰但越大包")]
        [Range(1, 100)]
        public int JpegQuality = 70;

        [Tooltip("UnityWebRequest timeout (seconds)")]
        public int RequestTimeoutSec = 90;

        [Tooltip("若截圖太大，可設定最大寬度做縮圖（0=不縮）。例如 960 或 1280")]
        public int MaxWidth = 0;

        [Header("UI")]
        public int UiPane = DebugUIBuilder.DEBUG_PANE_CENTER;

        [Tooltip("回覆框高度 = 按鈕高度 * 倍率（建議 4~7）")]
        public float AnswerBoxHeightMultiplier = 6f;

        [Tooltip("回覆框內距 (px)")]
        public float AnswerPadding = 12f;

        [Tooltip("若要強制更大字可填，例如 40；0 = 自動用按鈕字體大小")]
        public int AnswerFontSizeOverride = 0;

        private Text _statusText;   // 小狀態（可留著）
        private Text _answerText;   // 大回覆框文字
        private bool _busy;

        [Serializable]
        private class AnalyzeRequest
        {
            public string request_id;
            public string mode;       // "where" or "meaning"
            public string image_b64;  // base64(JPEG bytes)
        }

        [Serializable]
        private class AnalyzeResponse
        {
            public string status;      // "ok" / "unsure" / "error"
            public string answer;      // 最終顯示文字
            public string place_name;  // 可有可無
            public string pred_class;
            public float confidence;
        }

        private void Start()
        {
            BuildMenu();
        }

        private void BuildMenu()
        {
            var ui = DebugUIBuilder.Instance;

            ui.AddLabel("成大校園導覽（Passthrough）", UiPane);
            ui.AddDivider(UiPane);
            ui.AddLabel("請選擇你要問的問題：", UiPane);

            // 先建立按鈕，順便抓「按鈕字體大小」當作回覆框的字體基準
            var btnWhereRT = ui.AddButton("這裡是哪裡？", () =>
            {
                if (!_busy) StartCoroutine(CaptureAndSend("where"));
            }, -1, UiPane);

            var btnMeaningRT = ui.AddButton("這個景點有什麼意義？", () =>
            {
                if (!_busy) StartCoroutine(CaptureAndSend("meaning"));
            }, -1, UiPane);

            int buttonFontSize = GetButtonFontSize(btnWhereRT);
            float buttonW = btnWhereRT.rect.width;
            float buttonH = btnWhereRT.rect.height;

            ui.AddDivider(UiPane);

            // 小狀態字（保留）
            var statusRt = ui.AddLabel("狀態：待命", UiPane);
            _statusText = statusRt.GetComponent<Text>();

            // ✅ 新增「大回覆框」
            CreateBigAnswerBox(ui, UiPane, buttonW, buttonH, buttonFontSize);

            // 初始顯示
            SetAnswer("回覆會顯示在這裡。\n\n請按上方按鈕開始辨識。");
            ui.Show();
        }

        private int GetButtonFontSize(RectTransform buttonRT)
        {
            if (buttonRT == null) return 36;
            var t = buttonRT.GetComponentInChildren<Text>(true);
            if (t != null && t.fontSize > 0) return t.fontSize;
            return 36;
        }

        private void CreateBigAnswerBox(DebugUIBuilder ui, int pane, float buttonW, float buttonH, int buttonFontSize)
        {
            // 先用 AddLabel 拿到一個會被 DebugUIBuilder 排版的 RectTransform:contentReference[oaicite:1]{index=1}
            var containerRT = ui.AddLabel("", pane);

            // containerRT 原本有一個小字 Text（label prefab），我們把它關掉，改用自己的 UI 組
            var oldText = containerRT.GetComponent<Text>();
            if (oldText != null) oldText.enabled = false;

            // 設定回覆框大小：寬 = 按鈕寬， 高 = 按鈕高 * 倍率
            containerRT.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, buttonW);
            containerRT.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, buttonH * AnswerBoxHeightMultiplier);

            // 背景（框框）
            var bgGO = new GameObject("AnswerBG", typeof(RectTransform), typeof(Image));
            bgGO.transform.SetParent(containerRT, false);
            bgGO.transform.SetAsFirstSibling();

            var bgRT = bgGO.GetComponent<RectTransform>();
            bgRT.anchorMin = Vector2.zero;
            bgRT.anchorMax = Vector2.one;
            bgRT.offsetMin = Vector2.zero;
            bgRT.offsetMax = Vector2.zero;

            var bgImg = bgGO.GetComponent<Image>();
            // 半透明黑底（你要改色也可以）
            bgImg.color = new Color(0f, 0f, 0f, 0.65f);

            // 文字
            var textGO = new GameObject("AnswerText", typeof(RectTransform), typeof(Text));
            textGO.transform.SetParent(containerRT, false);

            var textRT = textGO.GetComponent<RectTransform>();
            textRT.anchorMin = Vector2.zero;
            textRT.anchorMax = Vector2.one;
            textRT.offsetMin = new Vector2(AnswerPadding, AnswerPadding);
            textRT.offsetMax = new Vector2(-AnswerPadding, -AnswerPadding);

            _answerText = textGO.GetComponent<Text>();
            _answerText.text = "";
            _answerText.alignment = TextAnchor.UpperLeft;
            _answerText.horizontalOverflow = HorizontalWrapMode.Wrap;
            _answerText.verticalOverflow = VerticalWrapMode.Overflow;
            _answerText.supportRichText = true;
            _answerText.color = Color.white;

            // ✅ 字體至少跟按鈕一樣大
            int targetSize = (AnswerFontSizeOverride > 0) ? AnswerFontSizeOverride : buttonFontSize;
            _answerText.fontSize = targetSize;

            // 字型沿用 label prefab 的字型（避免你專案沒指到字體）
            if (oldText != null && oldText.font != null)
            {
                _answerText.font = oldText.font;
                _answerText.fontStyle = oldText.fontStyle;
                _answerText.lineSpacing = oldText.lineSpacing;
            }
        }

        private void SetStatus(string msg)
        {
            if (_statusText != null) _statusText.text = msg;
            Debug.Log(msg);

            // 讓你至少看得到（把狀態也同步到大回覆框前面）
            if (_answerText != null && _busy)
            {
                _answerText.text = msg + "\n\n(等待回覆中…)";
            }
        }

        private void SetAnswer(string msg)
        {
            if (_answerText != null) _answerText.text = msg;
            Debug.Log(msg);
        }

        private IEnumerator CaptureAndSend(string mode)
        {
            _busy = true;
            SetStatus("狀態：擷取畫面中…");

            yield return new WaitForEndOfFrame();

            Texture2D shot = null;
            Texture2D resized = null;
            byte[] jpgBytes = null;

            try
            {
                shot = ScreenCapture.CaptureScreenshotAsTexture();
                if (shot == null)
                {
                    SetAnswer("擷取失敗：CaptureScreenshotAsTexture 回傳 null");
                    _busy = false;
                    yield break;
                }

                if (MaxWidth > 0 && shot.width > MaxWidth)
                {
                    int newW = MaxWidth;
                    int newH = Mathf.RoundToInt((float)shot.height * newW / shot.width);
                    resized = ResizeTexture(shot, newW, newH);
                    jpgBytes = ImageConversion.EncodeToJPG(resized, JpegQuality);
                }
                else
                {
                    jpgBytes = ImageConversion.EncodeToJPG(shot, JpegQuality);
                }
            }
            catch (Exception e)
            {
                SetAnswer($"擷取失敗：{e.Message}");
                _busy = false;
                if (shot != null) Destroy(shot);
                if (resized != null) Destroy(resized);
                yield break;
            }
            finally
            {
                if (shot != null) Destroy(shot);
                if (resized != null) Destroy(resized);
            }

            var reqObj = new AnalyzeRequest
            {
                request_id = Guid.NewGuid().ToString("N"),
                mode = mode,
                image_b64 = Convert.ToBase64String(jpgBytes)
            };

            string reqJson = JsonUtility.ToJson(reqObj);
            byte[] bodyRaw = Encoding.UTF8.GetBytes(reqJson);

            SetStatus("狀態：已送出，等待電腦運算回傳…");

            using (var uwr = new UnityWebRequest(AnalyzeUrl, "POST"))
            {
                uwr.uploadHandler = new UploadHandlerRaw(bodyRaw);
                uwr.downloadHandler = new DownloadHandlerBuffer();
                uwr.SetRequestHeader("Content-Type", "application/json");
                uwr.timeout = RequestTimeoutSec;

                yield return uwr.SendWebRequest();

                if (uwr.result != UnityWebRequest.Result.Success)
                {
                    SetAnswer($"連線失敗：{uwr.error}\n\n{uwr.downloadHandler.text}");
                    _busy = false;
                    yield break;
                }

                string respText = uwr.downloadHandler.text;
                string display = TryParseAnswer(respText);

                // ✅ 最終回覆丟到「大回覆框」
                SetAnswer(display);
            }

            _busy = false;
            if (_statusText != null) _statusText.text = "狀態：待命";
        }

        private string TryParseAnswer(string json)
        {
            try
            {
                var resp = JsonUtility.FromJson<AnalyzeResponse>(json);
                if (resp != null && !string.IsNullOrEmpty(resp.answer))
                {
                    if (!string.IsNullOrEmpty(resp.place_name))
                        return $"辨識：{resp.place_name}（{resp.confidence:0.00}）\n\n{resp.answer}";

                    return resp.answer;
                }
            }
            catch
            {
                // ignore
            }
            return json; // fallback: 顯示整包（debug 用）
        }

        private Texture2D ResizeTexture(Texture2D src, int newW, int newH)
        {
            var rt = RenderTexture.GetTemporary(newW, newH, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(src, rt);

            var prev = RenderTexture.active;
            RenderTexture.active = rt;

            var tex = new Texture2D(newW, newH, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, newW, newH), 0, 0);
            tex.Apply();

            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);

            return tex;
        }
    }
}
