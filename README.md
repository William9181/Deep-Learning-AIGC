# 🎬 IMDb Sentiment Analysis with BERT & PEFT (LoRA/IA3)

這是一個基於 **Streamlit** 的互動式深度學習實驗平台，旨在展示 **Parameter-Efficient Fine-Tuning (PEFT)** 技術在自然語言處理（NLP）任務中的應用。

本專案使用 **BERT-base** 模型針對 **IMDb 電影評論資料集** 進行情緒分類，並整合了 **LoRA** 與 **IA3** 兩種高效微調技術，提供從模型訓練、參數視覺化到即時推論的完整體驗。

## ✨ 專案功能 (Features)

1.  **📝 專案摘要 (Project Summary)**：自動生成技術報告，說明 PEFT 技術優勢。
2.  **🛠️ 線上訓練實驗室 (Training Lab)**：
    * 在網頁上自定義超參數 (Rank, Alpha, Epochs)。
    * 即時訓練客製化的 LoRA/IA3 模型。
    * 即時顯示訓練 Loss 與進度條。
3.  **🎬 模型推論演示 (Inference Demo)**：
    * 支援 LoRA、IA3 及使用者自訓練模型 (Custom Model) 的切換。
    * 提供信心指數 (Confidence) 與正負面機率條視覺化。
4.  **📊 參數量視覺化 (Parameter Visualization)**：
    * 使用 Plotly 繪製對數尺度 (Log Scale) 圖表。
    * 直觀展示 PEFT 技術如何將可訓練參數減少至原本的 0.05% ~ 0.27%。

## ⚙️ 安裝指南 (Installation)

建議使用 Python 3.8 或以上版本。請在終端機執行以下指令安裝所需套件：

```bash
pip install torch torchvision torchaudio
pip install transformers datasets peft accelerate evaluate scikit-learn
pip install streamlit pandas plotly tabulate
