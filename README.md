# Spam classifier demo (AIOT_HW3)

這個專案是為課程作業建立的簡單 Spam 偵測 Demo（TF-IDF + Logistic Regression baseline），同時包含 Streamlit 網頁介面讓你可以互動測試與展示結果。

## 主要功能
- 訓練：`src/ml/train_baseline.py` 使用 TF-IDF + LogisticRegression 訓練 baseline。會輸出 `out/model.joblib` 與 `out/metrics.json`。
- Web demo：`app/streamlit_app.py` 使用 Streamlit 提供互動介面，可單筆預測、批次上傳、顯示模型指標、top features、ROC 與 confusion matrix。
- 調試工具：`diagnostics/run_model_check.py` 可快速載入已儲存模型並對數則範例訊息做預測輸出。

## 檔案重點
- `src/ml/train_baseline.py`：訓練程式（可從遠端 dataset 拉取資料）。
- `app/streamlit_app.py`：Streamlit 應用。已新增側邊欄（sidebar）包含 threshold、模型統計與 dataset preview。
- `out/model.joblib`：已訓練的序列化模型（向量器 + 分類器 bundle）。注意：將二進位模型放在 repo 會使 repo 增大，部署時可選擇移除並改用 train-on-first-run 或外部下載。
- `out/metrics.json`：模型評估指標（accuracy、precision、recall、f1、roc_auc）。
- `diagnostics/run_model_check.py`：診斷腳本，方便在本地快速檢查模型行為與範例預測。

## 快速上手（本機測試 — Windows / PowerShell）
1. 建議使用虛擬環境（`python -m venv .venv`），並安裝需求：

```powershell
# 建立 venv（若還沒建立）
python -m venv .venv
# 啟用 venv
.venv\Scripts\Activate.ps1
# 安裝需求
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. 訓練模型（預設會下載公開 SMS spam dataset）：

```powershell
.venv\Scripts\python src\ml\train_baseline.py --output-dir out
```

訓練完成後會在 `out/` 看到 `model.joblib` 與 `metrics.json`。

3. 本機啟動 Streamlit（會自動載入 `out/model.joblib`）：

```powershell
.venv\Scripts\python -m streamlit run app\streamlit_app.py
# 或：streamlit run app/streamlit_app.py
```

4. 進入網頁後：
- 側邊欄（左側）可調整 classification threshold（預設 0.5），以及查看 sample dataset 統計（spam/ham 比例）。
- 主畫面可輸入範例訊息、按 Predict，並在 `Raw model outputs` 中看到原始 numeric 預測（0/1）與 decision score（如有）。

## 部署到 Streamlit Community Cloud
1. 前往 https://share.streamlit.io，用 GitHub 帳號登入並允許存取你的 repo（`g114093002/AIOT_HW3`）。
2. 建立新 app，選擇 branch `main`，填入 file path：`app/streamlit_app.py`，按 Deploy。
3. 注意：如果你不想把 `out/model.joblib` 提交到 repo，部署後可能找不到模型。兩個解法：
   - 在 repo 中加入 `out/model.joblib`（最簡單但 repo 會變大）；
   - 或實作「train-on-first-run」讓 app 第一次啟動時自動訓練模型（我可以幫你加）。

## Threshold 與預測行為說明
- 預設使用 probability >= threshold 判定 spam（threshold 可在側邊欄調整）。
- 如果你發現看似 spam 的訊息被判為 ham，請先看 `Raw model outputs` 中的 prob 值；把 threshold 調低可以提高召回（但可能產生更多偽陽性）。

## 診斷指令（快速檢查模型輸出）
在專案根目錄執行：

```powershell
.venv\Scripts\python diagnostics\run_model_check.py
```

輸出會列出 `BUNDLE_KEYS`、`metrics.json` 內容，並對幾則示例訊息輸出 probability 與 label，方便 debug。

## 常見問題與建議
- 若部署後結果不同步，確認遠端是否包含 `out/model.joblib`，或採用 train-on-first-run。  
- 若要提高模型召回，可考慮：調整 threshold、使用 class_weight 提高 spam 權重、或更換/調參模型並重新訓練（可在 `src/ml/train_baseline.py` 做改動）。

## 開發/CI / OpenSpec
- OpenSpec change proposals 與 validation 已納入 `openspec/changes/`（可用 `openspec validate` 本地檢查）。
- 我已加入一個 GitHub Actions workflow 來檢查 OpenSpec 變更與基本 python byte-compile（參考 `.github/workflows/openspec-validate.yml`）。

## 想要我幫忙的項目
- 若你要：
  - 我可以加上 train-on-first-run（自動在 app 啟動時訓練並保存模型），適合部署到 Streamlit Cloud 而不放二進位模型到 repo；
  - 我可以做 probability calibration 或重新訓練以改進 recall；
  - 我可以美化 UI 使其更接近你提到的示例站（側邊欄導航、更多統計卡片、PR curve等）。

請回覆你要我優先做哪件（例如：`train-on-first-run`、`calibrate`、或 `UI美化`），我會繼續實作並 push。

---
作者 / 聯絡：請於專案內留言或在 GitHub Repo 開 Issue 提問。
