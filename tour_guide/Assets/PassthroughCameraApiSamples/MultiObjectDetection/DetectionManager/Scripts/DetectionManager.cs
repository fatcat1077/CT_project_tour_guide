// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections;
using System.Collections.Generic;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Events;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class DetectionManager : MonoBehaviour
    {
        [SerializeField] private WebCamTextureManager m_webCamTextureManager;

        [Header("Controls configuration")]
        [SerializeField] private OVRInput.RawButton m_actionButton = OVRInput.RawButton.A;

        [Header("Ui references")]
        [SerializeField] private DetectionUiMenuManager m_uiMenuManager;

        [Header("Placement configureation")]
        [SerializeField] private GameObject m_spwanMarker;
        [SerializeField] private EnvironmentRayCastSampleManager m_environmentRaycast;
        [SerializeField] private float m_spawnDistance = 0.25f;
        [SerializeField] private AudioSource m_placeSound;

        [Header("Sentis inference ref")]
        [SerializeField] private SentisInferenceRunManager m_runInference;
        [SerializeField] private SentisInferenceUiManager m_uiInference;

        [Header("Mahjong classifier (second model)")]
        [SerializeField] private MahjongClassifierRunManager m_classifier;   // 第二個模型：牌面分類
        [SerializeField] private bool m_enableClassification = true;         // 是否啟用第二階段分類

        [Space(10)]
        public UnityEvent<int> OnObjectsIdentified;

        private bool m_isPaused = true;
        private List<GameObject> m_spwanedEntities = new();
        private bool m_isStarted = false;
        private bool m_isSentisReady = false;
        private float m_delayPauseBackTime = 0;

        #region Unity Functions
        private void Awake() => OVRManager.display.RecenteredPose += CleanMarkersCallBack;

        private IEnumerator Start()
        {
            // Wait until Sentis model is loaded (YOLO 模型)
            var sentisInference = FindAnyObjectByType<SentisInferenceRunManager>();
            while (!sentisInference.IsModelLoaded)
            {
                yield return null;
            }
            m_isSentisReady = true;
        }

        private void Update()
        {
            // Get the WebCamTexture CPU image
            var hasWebCamTextureData = m_webCamTextureManager.WebCamTexture != null;

            if (!m_isStarted)
            {
                // Manage the Initial Ui Menu
                if (hasWebCamTextureData && m_isSentisReady)
                {
                    m_uiMenuManager.OnInitialMenu(m_environmentRaycast.HasScenePermission());
                    m_isStarted = true;
                }
            }
            else
            {
                // Press A button to spawn 3d markers
                if (OVRInput.GetUp(m_actionButton) && m_delayPauseBackTime <= 0)
                {
                    SpwanCurrentDetectedObjects();
                }
                // Cooldown for the A button after return from the pause menu
                m_delayPauseBackTime -= Time.deltaTime;
                if (m_delayPauseBackTime <= 0)
                {
                    m_delayPauseBackTime = 0;
                }
            }

            // Not start a sentis inference if the app is paused or we don't have a valid WebCamTexture
            if (m_isPaused || !hasWebCamTextureData)
            {
                if (m_isPaused)
                {
                    // Set the delay time for the A button to return from the pause menu
                    m_delayPauseBackTime = 0.1f;
                }
                return;
            }

            // Run a new inference when the current inference finishes
            if (!m_runInference.IsRunning())
            {
                // ★ 在啟動下一輪 YOLO 前，用分類模型處理上一輪的框
                if (m_enableClassification && m_classifier != null)
                {
                    ClassifyCurrentDetections();
                }

                // 再啟動下一輪 YOLO 偵測
                m_runInference.RunInference(m_webCamTextureManager.WebCamTexture);
            }
        }
        #endregion

        #region Classification Functions

        /// <summary>
        /// 使用 MahjongClassifierRunManager 對目前 YOLO 偵測到的每個框進行分類，
        /// 並把結果寫回 BoundingBox.ClassName（UI 上就會顯示新的牌名），
        /// 同時輸出 Debug.Log。
        /// </summary>
        private void ClassifyCurrentDetections()
        {
            if (m_uiInference == null || m_uiInference.BoxDrawn == null)
                return;

            if (m_webCamTextureManager == null || m_webCamTextureManager.WebCamTexture == null)
                return;

            if (m_classifier == null || !m_classifier.IsModelLoaded)
                return;

            var camTex = m_webCamTextureManager.WebCamTexture;
            int imgW = camTex.width;
            int imgH = camTex.height;

            var boxes = m_uiInference.BoxDrawn;
            for (int i = 0; i < boxes.Count; i++)
            {
                var box = boxes[i];

                // 先記住 YOLO 原本的類別名稱（通常會是 "tile" 之類）
                string beforeClass = box.ClassName;

                // UI 座標 -> 影像 pixel 座標
                int w = Mathf.Clamp(Mathf.RoundToInt(box.Width), 1, imgW);
                int h = Mathf.Clamp(Mathf.RoundToInt(box.Height), 1, imgH);

                int cx = Mathf.RoundToInt(box.CenterX + imgW * 0.5f);
                int cy = Mathf.RoundToInt(imgH * 0.5f - box.CenterY); // Y 軸反轉

                int xMin = Mathf.Clamp(cx - w / 2, 0, imgW - 1);
                int yMin = Mathf.Clamp(cy - h / 2, 0, imgH - 1);
                int width = Mathf.Clamp(w, 1, imgW - xMin);
                int height = Mathf.Clamp(h, 1, imgH - yMin);

                // 從 WebCamTexture 裁出這個框
                Color[] pixels = camTex.GetPixels(xMin, yMin, width, height);
                Texture2D tileTex = new Texture2D(width, height, TextureFormat.RGB24, false);
                tileTex.SetPixels(pixels);
                tileTex.Apply();

                // 丟給分類器跑第二個模型
                m_classifier.RunInference(tileTex);
                string predictedLabel = m_classifier.GetLastResult();

                // 用完記得釋放暫存 Texture，避免累積記憶體
                Object.Destroy(tileTex);

                // 更新框的 ClassName（之後 3D marker / UI 會用到）
                box.ClassName = predictedLabel;
                // 若希望 UI 上只顯示牌名，可以同步改 label：
                box.Label = $"Class: {predictedLabel}";

                boxes[i] = box;   // BoundingBox 是 struct，要寫回 List 才會生效

                // ★ Debug：偵測＋分類結果輸出到 Log
                Debug.Log(
                    $"[MahjongClassifier] Detected tile #{i}: YOLO={beforeClass} → Classifier={predictedLabel}, " +
                    $"Pos=({xMin},{yMin}), Size=({width}x{height})"
                );
            }
            // ★ 新增：把更新過的 BoxDrawn 套用回 UI 上的 Text
            m_uiInference.RefreshBoxLabels(useClassNameOnly: true);
        }

        #endregion

        #region Marker Functions
        /// <summary>
        /// Clean 3d markers when the tracking space is re-centered.
        /// </summary>
        private void CleanMarkersCallBack()
        {
            foreach (var e in m_spwanedEntities)
            {
                Destroy(e, 0.1f);
            }
            m_spwanedEntities.Clear();
            OnObjectsIdentified?.Invoke(-1);
        }
        /// <summary>
        /// Spwan 3d markers for the detected objects
        /// </summary>
        private void SpwanCurrentDetectedObjects()
        {
            var count = 0;
            foreach (var box in m_uiInference.BoxDrawn)
            {
                if (PlaceMarkerUsingEnvironmentRaycast(box.WorldPos, box.ClassName))
                {
                    count++;
                }
            }
            if (count > 0)
            {
                // Play sound if a new marker is placed.
                m_placeSound.Play();
            }
            OnObjectsIdentified?.Invoke(count);
        }

        /// <summary>
        /// Place a marker using the environment raycast
        /// </summary>
        private bool PlaceMarkerUsingEnvironmentRaycast(Vector3? position, string className)
        {
            // Check if the position is valid
            if (!position.HasValue)
            {
                return false;
            }

            // Check if你 spanwed the same object before
            var existMarker = false;
            foreach (var e in m_spwanedEntities)
            {
                var markerClass = e.GetComponent<DetectionSpawnMarkerAnim>();
                if (markerClass)
                {
                    var dist = Vector3.Distance(e.transform.position, position.Value);
                    if (dist < m_spawnDistance && markerClass.GetYoloClassName() == className)
                    {
                        existMarker = true;
                        break;
                    }
                }
            }

            if (!existMarker)
            {
                // spawn a visual marker
                var eMarker = Instantiate(m_spwanMarker);
                m_spwanedEntities.Add(eMarker);

                // Update marker transform with the real world transform
                eMarker.transform.SetPositionAndRotation(position.Value, Quaternion.identity);
                eMarker.GetComponent<DetectionSpawnMarkerAnim>().SetYoloClassName(className);
            }

            return !existMarker;
        }
        #endregion

        #region Public Functions
        /// <summary>
        /// Pause the detection logic when the pause menu is active
        /// </summary>
        public void OnPause(bool pause)
        {
            m_isPaused = pause;
        }
        #endregion
    }
}
