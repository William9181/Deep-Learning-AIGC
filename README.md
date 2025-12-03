# 🎬 IMDb Sentiment Analysis with BERT & PEFT (LoRA/IA3)

這是一個基於 **Streamlit** 的全功能互動式深度學習實驗平台。本專案展示了如何利用 **Parameter-Efficient Fine-Tuning (PEFT)** 技術，在消費級硬體上高效微調 BERT 模型。

專案整合了 **LoRA** (Low-Rank Adaptation) 與 **IA3** 兩種主流技術，並提供完整的線上訓練、推論演示與參數視覺化分析功能。

---

## 📋 目錄 (Table of Contents)
- [專案特色](#-專案特色)
- [環境安裝](#-環境安裝)
- [快速開始](#-快速開始)
- [平台功能操作手冊](#-平台功能操作手冊)
- [檔案結構](#-檔案結構)
- [技術原理](#-技術原理)

---

## ✨ 專案特色

1.  **無需編寫程式碼的訓練體驗**：透過圖形化介面 (GUI) 調整 Rank, Alpha, Learning Rate 等超參數。
2.  **即時回饋機制**：訓練過程中即時顯示 Loss 變化與進度條。
3.  **多模型切換推論**：支援無縫切換 LoRA、IA3 或自定義訓練的模型進行情緒預測。
4.  **數據視覺化**：使用 Plotly 繪製對數尺度圖表，直觀呈現 PEFT 技術將參數量減少 1000 倍的震撼效果。
5.  **自動化摘要**：內建專案技術摘要頁面，方便報告使用。

---

## ⚙️ 環境安裝

建議使用 Python 3.8 或以上版本。

### 1. 複製專案
```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 2. 安裝依賴套件
請直接在終端機 (Terminal) 執行以下指令：

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)  # 若有 NVIDIA 顯卡 (選用)
pip install transformers datasets peft accelerate evaluate scikit-learn
pip install streamlit pandas plotly tabulate
```

> **注意**：若您沒有 NVIDIA 顯示卡，請省略第一行的 `--index-url` 部分，直接安裝 CPU 版 PyTorch 即可。

---

## 🚀 快速開始

### 步驟一：初始化模型與數據
在啟動 Web App 之前，我們需要先下載資料集並建立基礎的對照組模型（LoRA 與 IA3）。

請執行初始化腳本：

```bash
python compare_peft.py
```

**此腳本會執行以下動作：**
1. 下載 IMDb 資料集。
2. 自動訓練一個標準的 LoRA 模型與 IA3 模型。
3. 將訓練結果與參數統計輸出為 `peft_comparison_results.csv`。
4. 將模型權重儲存至 `bert-lora-imdb-final` 與 `bert-ia3-imdb-final`。

### 步驟二：啟動 Streamlit 應用程式
初始化完成後，執行以下指令啟動平台：

```bash
streamlit run app.py
```

成功後，您的預設瀏覽器會自動開啟以下網址：
> `http://localhost:8501`

---

## 📖 平台功能操作手冊

啟動 App 後，您可以在左側側邊欄切換以下四個功能頁面：

### 1. 📝 專案摘要
* 查看本專案的研究目的、技術架構與實驗結論。
* 適合用於簡報或快速了解專案背景。

### 2. 🛠️ 線上訓練實驗室 (Training Lab)
這是一個互動式的模型訓練場。
* **設定參數**：在介面上選擇 PEFT 方法 (LoRA/IA3)，並調整 Rank (矩陣秩)、Epochs (訓練輪數) 等。
* **開始訓練**：點擊 `🚀 開始訓練模型` 按鈕。
* **監控進度**：畫面會顯示即時進度條與 Loss 數值。
* **自動儲存**：訓練完成後，模型會自動儲存至 `custom_user_model` 資料夾，並自動註冊到推論頁面。

### 3. 🎬 模型推論演示 (Inference Demo)
在此頁面測試模型的實際表現。
* **選擇模型**：下拉選單可選擇預訓練的 LoRA/IA3 模型，或選擇 **"Custom Trained Model"** (您剛才在實驗室訓練的模型)。
* **輸入測試**：
    * 點擊「帶入正面/負面範例」快速測試。
    * 或自行輸入任何英文電影評論。
* **分析結果**：系統會顯示情緒判斷 (Positive/Negative)、信心指數 (Confidence) 以及詳細的機率分佈條。

### 4. 📊 參數量視覺化分析
* **Log Scale Chart**：透過對數尺度長條圖，對比 Full Fine-tuning (1.1億參數) 與 PEFT (僅數萬參數) 的巨大差異。
* **Linear Scale Chart**：詳細對比 LoRA 與 IA3 之間的參數效率差異。

---

## 📂 檔案結構

```text
Project_Root/
├── app.py                      # [核心] Streamlit Web App 主程式
├── compare_peft.py             # [核心] 初始化與批量訓練腳本
├── README.md                   # 說明文件
├── peft_comparison_results.csv # (Auto) 參數統計數據
├── custom_user_model/          # (Auto) 線上實驗室產出的模型
├── bert-lora-imdb-final/       # (Auto) 預設 LoRA 模型權重
└── bert-ia3-imdb-final/        # (Auto) 預設 IA3 模型權重
```

---

## 🧠 技術原理

本專案基於 Hugging Face `transformers` 與 `peft` 庫。

| 技術 | 核心概念 | 參數量變化 (約略) |
| :--- | :--- | :--- |
| **Full Fine-tuning** | 更新所有模型權重 | 100% (110M params) |
| **LoRA** | 凍結原權重，訓練旁路低秩矩陣 ($W + A \times B$) | ~0.27% (300K params) |
| **IA3** | 透過學習向量對激活值進行縮放 (Rescaling) | ~0.05% (60K params) |

---

## ⚠️ 故障排除 (Troubleshooting)

* **Q: 執行 `compare_peft.py` 時出現 `ModuleNotFoundError`?**
  * A: 請確認您已安裝所有依賴套件：`pip install -r requirements.txt` (或參考上方安裝指令)。

* **Q: 訓練時非常慢，且顯示 `UserWarning: no accelerator is found`?**
  * A: 這表示您正在使用 CPU 訓練。建議安裝 CUDA 版 PyTorch 以啟用 GPU 加速。若只能用 CPU，請在「線上訓練實驗室」將 `訓練樣本數` 調低至 50 以加速演示。

* **Q: App 顯示 `FileNotFoundError: 找不到模型資料夾`?**
  * A: 您必須先執行一次 `python compare_peft.py` 來生成必要的模型檔案。

---
**Created by William | 2025**
