// Copyright (c) Meta Platforms, Inc. and affiliates.
// Modified for: 2-button landmark QA + send frame to PC server + show answer (BIG BOX + Followups)

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
        public int RequestTimeoutSec = 180;

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

        private Text _statusText;   // 小狀態
        private Text _answerText;   // 大回覆框
        private bool _busy;

        // ✅ 追問按鈕（改到大回覆框最上方）
        private RectTransform _followBtn1RT;
        private RectTransform _followBtn2RT;
        private Button _followBtn1;
        private Button _followBtn2;
        private Text _followBtn1Text;
        private Text _followBtn2Text;
        private string _followQ1 = "";
        private string _followQ2 = "";

        // ✅ 記住最後一次送出的圖片（追問時用同一張）
        private byte[] _lastJpgBytes = null;

        [Serializable]
        private class AnalyzeRequest
        {
            public string request_id;
            public string mode;       // "where" | "meaning" | "both" | "ask"
            public string image_b64;  // base64(JPEG bytes)
            public string question;   // ✅ mode=ask 時使用
        }

        [Serializable]
        private class AnalyzeResponse
        {
            public string status;       // "ok" / "unsure" / "error"
            public string answer;       // 顯示文字
            public string[] followups;  // ✅ 兩個追問問題
            public string place_name;
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

            ui.AddLabel("麻將小幫手", UiPane);
            ui.AddDivider(UiPane);
            ui.AddLabel("請選擇你要問的問題：", UiPane);

            // 先建立按鈕，抓字體大小作為回覆框基準
            var btnWhereRT = ui.AddButton("切換進攻模式", () =>
            {
                if (!_busy) StartCoroutine(CaptureAndSend("where"));
            }, -1, UiPane);

            ui.AddButton("切換防守模式", () =>
            {
                if (!_busy) StartCoroutine(CaptureAndSend("meaning"));
            }, -1, UiPane);

            int buttonFontSize = GetButtonFontSize(btnWhereRT);
            float buttonW = btnWhereRT.rect.width;
            float buttonH = btnWhereRT.rect.height;

            ui.AddDivider(UiPane);

            // 小狀態
            var statusRt = ui.AddLabel("狀態：待命", UiPane);
            _statusText = statusRt.GetComponent<Text>();

            // 大回覆框（✅ 追問按鈕會被建在回覆框內最上方）
            CreateBigAnswerBox(ui, UiPane, buttonW, buttonH, buttonFontSize);

            // 初始化 disabled
            SetFollowups(null);

            SetAnswer("回覆會顯示在這裡。\n\n請按A按鈕開始辨識。");
            ui.Show();
        }

        private void OnFollowup1()
        {
            if (_busy) return;
            if (string.IsNullOrEmpty(_followQ1)) return;
            StartCoroutine(SendAskWithLastImage(_followQ1));
        }

        private void OnFollowup2()
        {
            if (_busy) return;
            if (string.IsNullOrEmpty(_followQ2)) return;
            StartCoroutine(SendAskWithLastImage(_followQ2));
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
            var containerRT = ui.AddLabel("", pane);

            var oldText = containerRT.GetComponent<Text>();
            if (oldText != null) oldText.enabled = false;

            containerRT.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, buttonW);
            containerRT.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, buttonH * AnswerBoxHeightMultiplier);

            // 背景
            var bgGO = new GameObject("AnswerBG", typeof(RectTransform), typeof(Image));
            bgGO.transform.SetParent(containerRT, false);
            bgGO.transform.SetAsFirstSibling();

            var bgRT = bgGO.GetComponent<RectTransform>();
            bgRT.anchorMin = Vector2.zero;
            bgRT.anchorMax = Vector2.one;
            bgRT.offsetMin = Vector2.zero;
            bgRT.offsetMax = Vector2.zero;

            var bgImg = bgGO.GetComponent<Image>();
            bgImg.color = new Color(0f, 0f, 0f, 0.65f);

            // ✅ 追問區：放在回覆框「最上方」
            float followBtnH = buttonH;       // 比照原本按鈕高度
            float followGap = 6f;
            float followPanelH = followBtnH * 2f + followGap;

            var followPanelGO = new GameObject("FollowupsPanel", typeof(RectTransform));
            followPanelGO.transform.SetParent(containerRT, false);

            var followPanelRT = followPanelGO.GetComponent<RectTransform>();
            followPanelRT.anchorMin = new Vector2(0f, 1f);
            followPanelRT.anchorMax = new Vector2(1f, 1f);
            followPanelRT.pivot = new Vector2(0.5f, 1f);
            followPanelRT.sizeDelta = new Vector2(0f, followPanelH);
            followPanelRT.anchoredPosition = new Vector2(0f, -AnswerPadding);

            // 字體/字級參考
            Font fontRef = (oldText != null) ? oldText.font : Resources.GetBuiltinResource<Font>("Arial.ttf");
            FontStyle fontStyleRef = (oldText != null) ? oldText.fontStyle : FontStyle.Normal;

            int baseSize = (AnswerFontSizeOverride > 0) ? AnswerFontSizeOverride : buttonFontSize;
            int followFontSize = Mathf.Max(18, baseSize - 6);

            // 追問按鈕 1
            _followBtn1RT = CreateFollowupButton(
                followPanelRT,
                "FollowupButton1",
                yTop: 0f,
                height: followBtnH,
                font: fontRef,
                fontStyle: fontStyleRef,
                fontSize: followFontSize,
                onClick: OnFollowup1,
                out _followBtn1,
                out _followBtn1Text
            );

            // 追問按鈕 2
            _followBtn2RT = CreateFollowupButton(
                followPanelRT,
                "FollowupButton2",
                yTop: followBtnH + followGap,
                height: followBtnH,
                font: fontRef,
                fontStyle: fontStyleRef,
                fontSize: followFontSize,
                onClick: OnFollowup2,
                out _followBtn2,
                out _followBtn2Text
            );

            // ✅ 回覆文字區：往下避開追問區
            var textGO = new GameObject("AnswerText", typeof(RectTransform), typeof(Text));
            textGO.transform.SetParent(containerRT, false);

            var textRT = textGO.GetComponent<RectTransform>();
            textRT.anchorMin = Vector2.zero;
            textRT.anchorMax = Vector2.one;

            float textTopReserved = AnswerPadding + followPanelH + 8f; // 追問區 + 間距
            textRT.offsetMin = new Vector2(AnswerPadding, AnswerPadding);
            textRT.offsetMax = new Vector2(-AnswerPadding, -textTopReserved);

            _answerText = textGO.GetComponent<Text>();
            _answerText.text = "";
            _answerText.alignment = TextAnchor.UpperLeft;
            _answerText.horizontalOverflow = HorizontalWrapMode.Wrap;
            _answerText.verticalOverflow = VerticalWrapMode.Overflow;
            _answerText.supportRichText = true;
            _answerText.color = Color.white;

            int targetSize = (AnswerFontSizeOverride > 0) ? AnswerFontSizeOverride : buttonFontSize;
            _answerText.fontSize = targetSize;

            if (oldText != null && oldText.font != null)
            {
                _answerText.font = oldText.font;
                _answerText.fontStyle = oldText.fontStyle;
                _answerText.lineSpacing = oldText.lineSpacing;
            }
            else
            {
                _answerText.font = fontRef;
                _answerText.fontStyle = fontStyleRef;
            }
        }

        private RectTransform CreateFollowupButton(
            RectTransform parent,
            string name,
            float yTop,
            float height,
            Font font,
            FontStyle fontStyle,
            int fontSize,
            UnityEngine.Events.UnityAction onClick,
            out Button btn,
            out Text txt)
        {
            var btnGO = new GameObject(name, typeof(RectTransform), typeof(Image), typeof(Button));
            btnGO.transform.SetParent(parent, false);

            var rt = btnGO.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0f, 1f);
            rt.anchorMax = new Vector2(1f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.sizeDelta = new Vector2(0f, height);
            rt.anchoredPosition = new Vector2(0f, -yTop);

            var img = btnGO.GetComponent<Image>();
            img.color = new Color(0f, 0f, 0f, 0.35f);

            btn = btnGO.GetComponent<Button>();
            btn.transition = Selectable.Transition.ColorTint;
            btn.onClick.RemoveAllListeners();
            btn.onClick.AddListener(onClick);

            // 文字
            var textGO = new GameObject("Text", typeof(RectTransform), typeof(Text));
            textGO.transform.SetParent(btnGO.transform, false);

            var textRT = textGO.GetComponent<RectTransform>();
            textRT.anchorMin = Vector2.zero;
            textRT.anchorMax = Vector2.one;
            float pad = 8f;
            textRT.offsetMin = new Vector2(pad, pad);
            textRT.offsetMax = new Vector2(-pad, -pad);

            txt = textGO.GetComponent<Text>();
            txt.text = "";
            txt.font = font;
            txt.fontStyle = fontStyle;
            txt.fontSize = fontSize;
            txt.alignment = TextAnchor.MiddleCenter;
            txt.horizontalOverflow = HorizontalWrapMode.Wrap;
            txt.verticalOverflow = VerticalWrapMode.Truncate;
            txt.supportRichText = true;
            txt.color = Color.white;

            return rt;
        }

        private void SetStatus(string msg)
        {
            if (_statusText != null) _statusText.text = msg;
            Debug.Log(msg);

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

        private void SetFollowups(string[] qs)
        {
            // 任何送出期間先禁用
            if (_followBtn1 != null) _followBtn1.interactable = false;
            if (_followBtn2 != null) _followBtn2.interactable = false;

            _followQ1 = "";
            _followQ2 = "";

            if (qs == null || qs.Length < 2 || string.IsNullOrEmpty(qs[0]) || string.IsNullOrEmpty(qs[1]))
            {
                if (_followBtn1Text != null) _followBtn1Text.text = "（建議打這張牌）";
                if (_followBtn2Text != null) _followBtn2Text.text = "（現在距離胡牌還有兩進聽）";
                return;
            }

            _followQ1 = qs[0];
            _followQ2 = qs[1];

            if (_followBtn1Text != null) _followBtn1Text.text = _followQ1;
            if (_followBtn2Text != null) _followBtn2Text.text = _followQ2;

            if (_followBtn1 != null) _followBtn1.interactable = true;
            if (_followBtn2 != null) _followBtn2.interactable = true;
        }

        private IEnumerator CaptureAndSend(string mode)
        {
            _busy = true;
            SetFollowups(null);
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

            // ✅ 記住最後一張
            _lastJpgBytes = jpgBytes;

            yield return SendRequest(mode, jpgBytes, question: "");
            _busy = false;
            if (_statusText != null) _statusText.text = "狀態：待命";
        }

        private IEnumerator SendAskWithLastImage(string question)
        {
            if (_lastJpgBytes == null)
            {
                SetAnswer("按下A鍵啟用手牌分析。");
                yield break;
            }

            _busy = true;
            SetFollowups(null);
            SetStatus("狀態：正在分析");

            yield return SendRequest("ask", _lastJpgBytes, question);

            _busy = false;
            if (_statusText != null) _statusText.text = "狀態：待命";
        }

        private IEnumerator SendRequest(string mode, byte[] jpgBytes, string question)
        {
            var reqObj = new AnalyzeRequest
            {
                request_id = Guid.NewGuid().ToString("N"),
                mode = mode,
                image_b64 = Convert.ToBase64String(jpgBytes),
                question = question
            };

            string reqJson = JsonUtility.ToJson(reqObj);
            byte[] bodyRaw = Encoding.UTF8.GetBytes(reqJson);

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
                    SetFollowups(null);
                    yield break;
                }

                string respText = uwr.downloadHandler.text;
                HandleResponse(respText);
            }
        }

        private void HandleResponse(string json)
        {
            try
            {
                var resp = JsonUtility.FromJson<AnalyzeResponse>(json);
                if (resp != null && !string.IsNullOrEmpty(resp.answer))
                {
                    string header = "";
                    if (!string.IsNullOrEmpty(resp.place_name))
                        header = $"辨識：{resp.place_name}（{resp.confidence:0.00}）\n\n";

                    SetAnswer(header + resp.answer);

                    // ✅ 更新追問按鈕
                    if (resp.followups != null && resp.followups.Length >= 2)
                        SetFollowups(resp.followups);
                    else
                        SetFollowups(null);

                    return;
                }
            }
            catch
            {
                // ignore
            }

            // fallback：顯示原文（debug 用）
            SetAnswer(json);
            SetFollowups(null);
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
