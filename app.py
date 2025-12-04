import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
# ✅ 修改後的寫法 (拆開來匯入)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
# 關鍵：直接從子模組匯入 Trainer，避開 transformers 主入口的檢查
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from datasets import load_dataset
from peft import PeftModel, LoraConfig, IA3Config, TaskType, get_peft_model
import os
import shutil
import time

# 設定頁面配置
st.set_page_config(page_title="IMDb PEFT Project", page_icon="🎬", layout="wide")

# 初始化 Session State
if "custom_model_path" not in st.session_state:
    st.session_state["custom_model_path"] = None
if "custom_model_name" not in st.session_state:
    st.session_state["custom_model_name"] = ""

# --- 側邊欄導航 ---
st.sidebar.title("功能導航")
app_mode = st.sidebar.radio(
    "選擇功能頁面", 
    ["📝 專案摘要", "🛠️ 線上訓練實驗室", "🎬 模型推論演示", "📊 參數量視覺化分析"]
)

# ===========================
# 工具類別: Streamlit 訓練回調
# ===========================
class StreamlitLogCallback(TrainerCallback):
    def __init__(self, progress_bar, status_text, total_steps):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_steps = total_steps

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        if self.total_steps > 0:
            progress = min(current_step / self.total_steps, 1.0)
            self.progress_bar.progress(progress)
            loss_log = "..."
            if state.log_history:
                for log in reversed(state.log_history):
                    if "loss" in log:
                        loss_log = f"{log['loss']:.4f}"
                        break
            self.status_text.text(f"Training... Step {current_step}/{self.total_steps} (Loss: {loss_log})")

# ===========================
# 共用函式: 模型載入
# ===========================
@st.cache_resource
def load_model_pipeline(method_name, custom_path=None):
    base_model_name = "bert-base-uncased"
    if method_name == "Custom Trained Model":
        peft_model_id = custom_path
    else:
        model_paths = {
            "LoRA": "bert-lora-imdb-final",
            "IA3": "bert-ia3-imdb-final"
        }
        peft_model_id = model_paths.get(method_name)
    
    if not peft_model_id or not os.path.exists(peft_model_id):
        if method_name in ["LoRA", "IA3"]:
            raise FileNotFoundError(f"找不到模型資料夾：{peft_model_id}。請先執行 compare_peft.py。")
        else:
            raise FileNotFoundError(f"找不到自定義模型路徑：{peft_model_id}")

    with st.spinner(f"正在載入 {method_name}..."):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"}, label2id={"NEGATIVE": 0, "POSITIVE": 1}
        )
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    return tokenizer, model, device

# ===========================
# 頁面 1: 專案摘要 (新增頁面)
# ===========================
if app_mode == "📝 專案摘要":
    st.title("📝 專案成果摘要")
    st.markdown("### 高效參數微調 (PEFT) 於 IMDb 情緒分析之應用")
    st.markdown("---")
    
    # 使用 Container 讓排版更漂亮
    with st.container():
        st.markdown("""
        #### 📌 專案概述
        本專案旨在探討 **高效參數微調技術 (Parameter-Efficient Fine-Tuning, PEFT)** 在自然語言處理任務中的應用效益。我們選用 **BERT-base** 預訓練模型，針對 **IMDb 電影評論資料集** 進行二元情緒分類（正面/負面）任務。
        
        #### 🧪 技術與方法
        實驗核心在於對比傳統的「全量微調 (Full Fine-tuning)」與兩種主流 PEFT 技術：
        * **LoRA (Low-Rank Adaptation)**：透過低秩矩陣分解，僅訓練旁路參數。
        * **IA3**：透過抑制與放大內部激活值向量進行微調。
        
        #### 📊 實驗結論
        實驗結果證實，在資源受限的環境下，PEFT 技術展現了極大的優勢：
        1.  **極致輕量化**：相較於全量微調（約 1.1 億參數），**LoRA 僅需訓練約 0.27%** 的參數，而 **IA3 更進一步降至 0.05%**，大幅降低了儲存與記憶體需求。
        2.  **效能優異**：在大幅減少參數量的情況下，PEFT 模型仍能達到與全量微調相當的預測準確率。
        
        #### 💻 平台功能
        本系統整合 Streamlit 構建了完整的互動式實驗平台，包含：
        * **線上訓練實驗室**：允許使用者即時調整參數（如 Rank, Learning Rate）並訓練客製化模型。
        * **模型推論演示**：提供即時文本輸入與情緒信心指數分析。
        * **視覺化分析**：利用對數尺度圖表，直觀呈現參數量數個數量級的縮減差異。
        """)
        
    st.info("👈 請點擊左側導航欄位，開始體驗各項功能。")

# ===========================
# 頁面 2: 線上訓練實驗室
# ===========================
elif app_mode == "🛠️ 線上訓練實驗室":
    st.title("🛠️ 線上訓練實驗室")
    st.markdown("在此頁面，您可以調整參數並**即時訓練**一個輕量級模型。")
    st.info("💡 提示：此實驗室使用極少量數據 (Sample) 進行快速演示。")
    
    col_conf, col_param = st.columns(2)
    with col_conf:
        st.subheader("1. 模型架構設定")
        peft_type = st.selectbox("選擇 PEFT 方法", ["LoRA", "IA3"])
        if peft_type == "LoRA":
            r_rank = st.slider("LoRA Rank (r)", 4, 32, 8, 4)
            lora_alpha = st.slider("LoRA Alpha", 8, 64, 16, 8)
            dropout = st.slider("Dropout", 0.0, 0.5, 0.1)
        else:
            st.info("IA3 不需要設定 Rank。")
            r_rank = 0
            
    with col_param:
        st.subheader("2. 訓練參數設定")
        lr_default = 2e-4 if peft_type=="LoRA" else 5e-3
        learning_rate = st.number_input("Learning Rate", value=lr_default, format="%.5f")
        epochs = st.slider("Epochs", 1, 5, 1)
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=0)
        sample_size = st.slider("訓練樣本數", 20, 200, 50)

    st.markdown("---")
    if st.button("🚀 開始訓練模型", type="primary"):
        status_area = st.empty()
        progress_bar = st.progress(0)
        try:
            with st.spinner("正在初始化..."):
                model_checkpoint = "bert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
                dataset = load_dataset("imdb")
                small_train = dataset["train"].shuffle(seed=int(time.time())).select(range(sample_size))
                
                def preprocess_function(examples):
                    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
                
                tokenized_train = small_train.map(preprocess_function, batched=True)
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
                base_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
                
                if peft_type == "LoRA":
                    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=r_rank, lora_alpha=lora_alpha, lora_dropout=dropout, target_modules=["query", "value"])
                else:
                    peft_config = IA3Config(task_type=TaskType.SEQ_CLS, target_modules=["key", "value", "output.dense"], feedforward_modules=["output.dense"])
                
                model = get_peft_model(base_model, peft_config)
                output_dir = f"./custom_trained_{peft_type}"
                total_steps = (len(small_train) // batch_size) * epochs
                if total_steps == 0: total_steps = epochs
                
                training_args = TrainingArguments(
                    output_dir=output_dir, learning_rate=learning_rate, per_device_train_batch_size=batch_size,
                    num_train_epochs=epochs, weight_decay=0.01, logging_steps=1, save_strategy="no", report_to="none",
                    use_cpu=not torch.cuda.is_available()
                )
                
                trainer = Trainer(
                    model=model, args=training_args, train_dataset=tokenized_train, tokenizer=tokenizer,
                    data_collator=data_collator, callbacks=[StreamlitLogCallback(progress_bar, status_area, total_steps)]
                )

            status_area.text("Training started...")
            trainer.train()
            save_path = "./custom_user_model"
            if os.path.exists(save_path): shutil.rmtree(save_path)
            model.save_pretrained(save_path)
            st.session_state["custom_model_path"] = save_path
            st.session_state["custom_model_name"] = f"Custom {peft_type} (Sample={sample_size}, Epochs={epochs})"
            progress_bar.progress(100)
            status_area.success(f"訓練完成！請前往「🎬 模型推論演示」頁面測試。")
            st.balloons()
        except Exception as e:
            st.error(f"錯誤: {e}")

# ===========================
# 頁面 3: 模型推論演示
# ===========================
elif app_mode == "🎬 模型推論演示":
    st.title("🎬 IMDb 電影評論情緒分析")
    st.markdown("---")
    st.sidebar.header("推論模型選擇")
    available_methods = []
    if os.path.exists("bert-lora-imdb-final"): available_methods.append("LoRA")
    if os.path.exists("bert-ia3-imdb-final"): available_methods.append("IA3")
    if st.session_state["custom_model_path"] and os.path.exists(st.session_state["custom_model_path"]):
        available_methods.append("Custom Trained Model")
        st.sidebar.success("✅ 偵測到您剛訓練的模型！")

    if not available_methods:
        st.error("找不到任何可用模型。請先執行 compare_peft.py 或在線上實驗室訓練。")
        st.stop()

    selected_method = st.sidebar.selectbox("選擇微調模型", available_methods)
    if selected_method == "Custom Trained Model":
        st.sidebar.caption(f"參數: {st.session_state['custom_model_name']}")

    try:
        custom_path = st.session_state["custom_model_path"] if selected_method == "Custom Trained Model" else None
        tokenizer, model, device = load_model_pipeline(selected_method, custom_path)
        st.sidebar.info(f"裝置: {device.upper()}")
    except Exception as e:
        st.error(f"載入失敗: {e}")
        st.stop()

    if "review_input" not in st.session_state: st.session_state["review_input"] = ""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("帶入正面範例"): st.session_state["review_input"] = "The cinematography was breathtaking and the story was deeply moving. I loved every minute of it!"
    with col2:
        if st.button("帶入負面範例"): st.session_state["review_input"] = "The plot made no sense and the acting was wooden. Total waste of time."

    user_input = st.text_area("請輸入影評：", height=150, key="review_input")
    if st.button("開始分析", type="primary") and user_input.strip():
        with st.spinner("分析中..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad(): outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred_id = outputs.logits.argmax().item()
            label = model.config.id2label[pred_id]
            confidence = probs[0][pred_id].item()
            prob_neg, prob_pos = probs[0][0].item(), probs[0][1].item()

        if label == "POSITIVE": st.success(f"🎉 正面 (信心指數: {confidence:.2%})")
        else: st.error(f"😞 負面 (信心指數: {confidence:.2%})")
        
        c1, c2 = st.columns(2)
        with c1: st.write("**Negative**"); st.progress(prob_neg); st.caption(f"{prob_neg:.1%}")
        with c2: st.write("**Positive**"); st.progress(prob_pos); st.caption(f"{prob_pos:.1%}")

# ===========================
# 頁面 4: 參數量視覺化分析
# ===========================
elif app_mode == "📊 參數量視覺化分析":
    st.title("📊 PEFT 參數量瘦身成果展示")
    st.markdown("---")
    csv_file = "peft_comparison_results.csv"
    if not os.path.exists(csv_file):
        st.warning(f"找不到數據文件 {csv_file}。請確保您已成功執行 compare_peft.py。")
    else:
        df = pd.read_csv(csv_file)
        df["Formatted Params"] = df["Trainable Params"].apply(lambda x: f"{x:,}")
        
        st.subheader("1. 可訓練參數量對比 (對數尺度)")
        fig_log = px.bar(df, x="Method", y="Trainable Params", color="Method", text="Formatted Params",
            title="Trainable Parameters (Log Scale)", log_y=True,
            color_discrete_map={"Full Fine-tuning (Baseline)": "lightgrey", "LoRA": "#636EFA", "IA3": "#EF553B"}, height=500)
        fig_log.update_traces(textposition='outside')
        st.plotly_chart(fig_log, use_container_width=True)

        st.subheader("2. PEFT 方法內部對決")
        df_peft = df[df["Method"] != "Full Fine-tuning (Baseline)"]
        if not df_peft.empty:
            fig_linear = px.bar(df_peft, x="Trainable Params", y="Method", color="Method", text="Formatted Params",
                title="LoRA vs. IA3 (Linear Scale)", orientation='h',
                color_discrete_map={"LoRA": "#636EFA", "IA3": "#EF553B"}, height=400)
            fig_linear.update_traces(textposition='outside')
            st.plotly_chart(fig_linear, use_container_width=True)
        

        st.markdown("### 詳細數據"); st.dataframe(df)
