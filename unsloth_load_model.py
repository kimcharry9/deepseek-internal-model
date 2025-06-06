from datasets import load_dataset
from unsloth import FastLanguageModel, to_sharegpt, standardize_sharegpt, get_chat_template,  apply_chat_template, is_bfloat16_supported, train_on_responses_only
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datetime import datetime

model_dir = "/data/deepseek-model/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
data_dir = "/data/deepseek-model/cleansed_dataset"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True  # หรือ False ถ้าคุณมี VRAM มากพอ
)

# Define some "Parameter Efficient Fine Tuning (PEFT)" to make less computation but still more efficiency.
# In this code, We will use "Low-Rank Adaptation (LoRA)" for purposed.
model = FastLanguageModel.get_peft_model(
    model,
    r = 4,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None
)

dataset = load_dataset("json", data_files=f"{data_dir}/system_disk_usage_json3_1m_trim.jsonl", split="train")
print(dataset.column_names)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1"
)

def formatting_prompts_func(dataset):
    convos = dataset['conversations']
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts }

dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset[0]['conversations'])
print(dataset[0]['text'])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    max_seq_length = 2048,
    dataset_num_proc = 4,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 10,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

trainer_stats = trainer.train()

current_date = datetime.now().strftime("%Y%m%d-%H%M%S")
trainer_stats_filename = f"trainer-stats-{current_date}.txt"

with open(f"{model_dir}/trainer-stats/{trainer_stats_filename}", "w") as file:
    file.write(str(trainer_stats))

print(f"Trainer stats saved to {model_dir}/trainer-stats/{trainer_stats_filename}")

# no-container model saving.
model.save_pretrained(f"{model_dir}/pretrained-model")
tokenizer.save_pretrained(f"{model_dir}/pretrained-model")

# container-based model saving. GGUF usage
model.save_pretrained_gguf(f"{model_dir}/pretrained-model", tokenizer)
