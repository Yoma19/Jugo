import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# -----------------------------
# Settings
# -----------------------------
MODEL_NAME = "lora-chat-out"   # Pretrained base model name or path
DATA_FILE = "Jugo_Training_Data.jsonl"
OUTPUT_DIR = "Jugo_LoRA_Model_1.1"      # Update every time you train
USE_4BIT = True

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token is None:                     #new
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Convert chat messages → prompt text
# -----------------------------
def format_chat(example):
    """
    Expected format:
    {
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
    """
    messages = example["messages"]
    if tokenizer.chat_template:
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback to Zephyr format
        formatted_text = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted_text += f"<|user|>\n{msg['content']}</s>\n"
            elif msg["role"] == "assistant":
                formatted_text += f"<|assistant|>\n{msg['content']}</s>\n"
        full_prompt = formatted_text

    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=2048,
        padding=False,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.map(format_chat, remove_columns=dataset.column_names)


# -----------------------------
# Load base model
# -----------------------------
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use nf4 for better performance
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Double quantization to save more memory
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)


# -----------------------------
# LoRA config
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# -----------------------------
# Training
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,     # 1 epoch for occasional training, 3 for more serious
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,  # Important for our custom formatting
    gradient_checkpointing=True,  # Enable gradient checkpointing
    optim="paged_adamw_8bit",     # Use 8-bit Adam optimizer to save memory
    report_to="none",             # Disable wandb if not needed
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8
        ),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished! Model saved to:", OUTPUT_DIR)
