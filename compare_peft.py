import torch
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, IA3Config, TaskType
import evaluate

# --- 1. 全局設定 ---
model_checkpoint = "bert-base-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"
results = [] # 儲存比較結果

# 載入資料 (為了快速實驗，我們只取少量樣本)
# 若要看完整效果，請將 .select(...) 移除
raw_dataset = load_dataset("imdb")
dataset = raw_dataset.map(
    lambda x: AutoTokenizer.from_pretrained(model_checkpoint)(
        x["text"], truncation=True, padding="max_length", max_length=128
    ), 
    batched=True
)
small_train = dataset["train"].shuffle(seed=42).select(range(500)) 
small_test = dataset["test"].shuffle(seed=42).select(range(100))

print(f"Dataset prepared. Train size: {len(small_train)}, Test size: {len(small_test)}")

# --- 2. 訓練評估函數 ---
def run_experiment(method_name, peft_config):
    print(f"\n{'='*20} Running {method_name} {'='*20}")
    
    # 每次都重新載入乾淨的 BERT
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # 應用 PEFT 設定
    model = get_peft_model(model, peft_config)
    
    # 計算參數量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    param_ratio = trainable_params / all_params
    
    print(f"Trainable Params: {trainable_params:,} ({param_ratio:.4%})")

    # 設定 Trainer
    training_args = TrainingArguments(
        output_dir=f"./results_{method_name}",
        learning_rate=5e-3 if method_name == "IA3" else 2e-4, # IA3 通常需要較大的 LR
        per_device_train_batch_size=16,
        num_train_epochs=3,
        eval_strategy="epoch", # 新版 transformers 請用 eval_strategy
        save_strategy="no",          # 實驗用，暫不存檔以節省空間
        logging_steps=50,
        report_to="none"             # 不上傳到 wandb
    )

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train,
        eval_dataset=small_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 計時並訓練
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    # 取得最終評估結果
    eval_result = trainer.evaluate()
    save_path = f"./bert-{method_name.lower()}-imdb-final"
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    # 紀錄數據
    results.append({
        "Method": method_name,
        "Accuracy": eval_result["eval_accuracy"],
        "Trainable Params": trainable_params,
        "Param %": f"{param_ratio:.4%}",
        "Training Time (s)": round(end_time - start_time, 2)
    })
    
    # 清理記憶體
    del model, trainer
    torch.cuda.empty_cache()

# --- 3. 定義不同的 Config ---

# Config A: LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    r=8, 
    lora_alpha=16, 
    target_modules=["query", "value"], 
    lora_dropout=0.1
)

# Config B: IA3
# IA3 針對 Transformer 通常會作用在 Key, Value 和 FeedForward 層
ia3_config = IA3Config(
    task_type=TaskType.SEQ_CLS,
    target_modules=["key", "value", "output.dense"], 
    feedforward_modules=["output.dense"]
)

# --- 4. 執行比較 ---
run_experiment("LoRA", lora_config)
run_experiment("IA3", ia3_config)

# --- 5. 顯示結果表格 ---
print("\n" + "="*40)
print("FINAL COMPARISON RESULTS")
print("="*40)
df = pd.DataFrame(results)
try:
    print(df.to_markdown(index=False))
except ImportError:
    print(df)

# --- [新增關鍵步驟] 儲存結果為 CSV 供 Streamlit 讀取 ---
csv_filename = "peft_comparison_results.csv"
# 為了視覺化，我們需要加入 BERT Base 的總參數量當作基準
bert_total_params = 109482240  # bert-base-uncased 的約略總參數量
baseline_row = {
    "Method": "Full Fine-tuning (Baseline)",
    "Accuracy": None, # 這裡不重要，只是為了比較參數
    "Trainable Params": bert_total_params,
    "Param %": "100.00%",
    "Training Time (s)": None
}
# 將基準線加入 DataFrame 最前面
df_final = pd.concat([pd.DataFrame([baseline_row]), df], ignore_index=True)

df_final.to_csv(csv_filename, index=False)
print(f"\nResults saved to {csv_filename} for visualization.")