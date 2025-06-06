# DeepSeek-R1: LLM with Fine-Tuning Implementation Guidelines

[TOC]

### 1. Import targeted libraries

```python
from datasets import load_dataset
from unsloth import FastLanguageModel, to_sharegpt, standardize_sharegpt, get_chat_template,  apply_chat_template, is_bfloat16_supported, train_on_responses_only
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datetime import datetime
```



### 2. [Optional] Define general variables

```python
model_dir = "/data/deepseek-model/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
data_dir = "/data/deepseek-model/cleansed_dataset"
```



### 3. Register targeted model 

```python
# Model Registering -- Choose your targeted model that you want.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True
)
```

- **model_name:** ชื่อโมเดลที่ต้องการใช้งาน โดยดึงมาใช้จาก Huggingface Hub (โดยปกติถ้าลง Unsloth Library มาแล้ว จะสามารถดึงโมเดลผ่านการกำหนด path แบบตัวอย่างได้เลย ไม่จำเป็นต้อง login ผ่าน Huggingface Hub)
- **max_seq_length:** จำนวน input token ที่โมเดลสามารถรับมาประมวลผลได้ในคราวเดียว ค่าที่สูงขึ้นจะใช้ VRAM มากขึ้น เพราะต้องเก็บค่า hidden states ในการคำนวณ 
- **dtype:** ประเภทของค่าตัวเลขในการประมวลผล ปกติจะใช้ None เพื่อให้โปรแกรมตรวจจับประเภทข้อมูลที่เหมาะสมกับ Hardware นั้น ๆ เอง
  - float16 ใช้ได้กับ Tesla T4, V100
  - Bfloat16 ใช้ได้กับ Ampere+, A100, H100  

- **load_in_4bit:** ใช้เทคนิค **4-bit quantization** เพื่อลดขนาดของโมเดล



### 4. [Fine-Tuning Pretrained Model with LoRA](https://arxiv.org/pdf/2106.09685)

```python
# Define some "Parameter Efficient Fine Tuning (PEFT)" to make less computation but still more efficiency.
# In this code, We will use "Low-Rank Adaptation (LoRA)" for purposed.
model = FastLanguageModel.get_peft_model(
    model,
    r = 4,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],              
    lora_alpha = 16,
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None
)
```

- **r:** Rank Decomposition กำหนด rank ว่าจะปรับลดพารามิเตอร์ที่ใช้คำนวณลงที่ขนาดเท่าไหร่ โดยปกติถ้าไม่ปรับจูน โมเดลทั้งหมดจะมีการคำนวณขนาดใหญ่มาก (Matrix Computation Based) ทำให้ใช้ resource มหาศาล

  - แทนที่จะอัปเดตค่าคำนวณโมเดลโดยตรง LoRA ใช้การประมาณค่าแบบ **Low-Rank Decomposition** โดยคำนวณจากสมการ

    ​											`ΔW ≈ BA`

    โดยที่

    - ΔW คือการเปลี่ยนแปลงของเมทริกซ์น้ำหนัก
    - B เป็นเมทริกซ์ขนาด d×r
    - A เป็นเมทริกซ์ขนาด r×d
    - ตัวอย่างเช่น
      - LLaMA 7B มี dmodel=4096
      - ดังนั้น โมเดลคำนวณ มีขนาด **4096×4096**
      - ถ้าใช้ LoRA กับ r=4 → **เพิ่มพารามิเตอร์แค่ 4096×4+4×4096 เท่านั้น**

- **target_modules:** กำหนดให้ LoRA ปรับแต่งเฉพาะบางพารามิเตอร์ใน **Self-Attention Mechanism**
  - `q_proj` (Query Projection)
  - `k_proj` (Key Projection)
  - `v_proj` (Value Projection)
  - `o_proj` (Output Projection)
  - `gate_proj` `up_proj` `down_proj` ([Feedforward Layer](https://pytorch.org/torchtune/0.4/generated/torchtune.modules.FeedForward.html)) 

- **[lora_alpha](https://medium.com/@drishtisharma96505/analyzing-the-impact-of-lora-alpha-on-llama-2-quantized-with-gptq-f01e8e8ed8fd):** scaling factor ที่ช่วยคุมผลกระทบการคำนวณ LoRA Matrix
  - ค่านี้จะไม่ค่อยคงที่ว่าควรใช้ค่าอะไร จำเป็นต้องทดสอบด้วยตนเอง หากค่าไหนที่ส่งผลให้เกิด loss น้อยสุด ให้ใช้ค่านั้น
  - โดยปกติค่าที่เหมาะสมจะอยู่ที่ 16 และ 32
  - ทั้งนี้ทั้งนั้น อาจขึ้นอยู่กับการกำหนดค่า r ด้วย

- **lora_dropout:** ค่าสำหรับป้องกันไม่ให้โมเดล Overfitting โดยการ "ปิด" หรือ "drop" หน่วยประมวลผล (neurons) ในเครือข่ายประสาท (neural network) แบบสุ่มในแต่ละรอบการฝึก โดยปกติจะใช้อยู่ที่ไม่เกิน 0.5
- **bias:** กำหนดว่าให้โมเดลมี Bias Term เพื่อช่วยให้การคำนวณโมเดลมีประสิทธิภาพขึ้นหรือไม่
  - ในโมเดล Deep Learning แต่ละ **Linear Layer** มักมีสมการ: `Y = XW + b` โดยที่
    - X = อินพุต
    - W = เมทริกซ์น้ำหนัก
    - b = **Bias Term** (ค่าคงที่ที่เพิ่มเข้าไป)
    - ปกติ LoRA จะอัปเดตเฉพาะ W เท่านั้น แต่ถ้ากำหนด `bias = "all"` LoRA จะอัปเดต b ด้วย ซึ่งอาจช่วยให้โมเดลเรียนรู้ได้ดีขึ้นในบางกรณี
- **use_gradient_checkpointing:** ช่วยลดการใช้ VRAM โดยคำนวณ Gradient ใหม่ในช่วง Backpropagation
- **random_state:** สำหรับกำหนดค่าตั้งต้น (seed) สำหรับสร้างตัวเลขสุ่ม โดยตัวเลขสุ่มที่ได้แต่ละชุดจะมีตัวเลขเดียวกันเสมอ โดยเลข 42 คือ magic number (ไม่มีทฤษฎีตายตัว แต่เกิดจากเรื่องเล่าต่าง ๆ รอบตัว)
  - เช่น สุ่มข้อมูล 1-100 มา 5 ตัว ด้วย random_state = 42 จะได้ข้อมูล [52 93 15 72 61] แบบนี้เสมอ 
  - เช่น สุ่มข้อมูล 1-100 มา 5 ตัว ด้วย random_state = 1 จะได้ข้อมูล [38 13 73 10 76] แบบนี้เสมอ
- **use_rslora:** **Rank Stabilized LoRA (RS-LoRA)** เป็นเทคนิคใหม่ที่ช่วยให้ LoRA มีความเสถียรมากขึ้น ถ้าโมเดลมีขนาดใหญ่หรือมีการเปลี่ยนแปลงเยอะ อาจตั้งเป็น `True` เพื่อเพิ่มเสถียรภาพ
  - โดยปกติถ้าอิงจากการกำหนดค่า r อย่างเดียว กับโมเดลขนาดใหญ่มาก หรือข้อมูลที่หลากหลายมาก การเปลี่ยนแปลงใน LoRA matrix อาจไม่สอดคล้องกับโมเดลหลัก ทำให้ค่า Loss ผันผวน (ความแม่นยำได้ไม่ดีเท่าที่ควร)
- **loftq_config:** ใช้เทคนิค **quantization** เพื่อลดขนาดของการคำนวณผ่าน LoRA อีกทีหนึ่ง เป็นคนละส่วนกันกับการทำ Quantization จากตัวโมเดล



### 5. Import targeted datasets

```python
dataset = load_dataset("json", data_files=f"{data_dir}/system_disk_usage_json3_1m_trim.jsonl", split="train")
print(dataset.column_names)
```

ตัวอย่าง datasets ที่ควรจะเป็น *(เนื่องจากการนำข้อมูลไปเทรนกับ pre-trained model จำเป็นต้องใช้ข้อมูลที่มี type หรือ pattern ที่สอดคล้องกับ datasets ต้นแบบของโมเดลนั้น ๆ  ดังนั้น ควรจะแปลงข้อมูลให้เป็นตาม pattern ที่โมเดลรองรับได้ก่อน)*

```json
{"conversations": [{"from": "system", "value": "You are an expert System Engineer, you will predict how much that our company has used CPU, Memory (GB) and Disk Usage (%) with our company data with really good assisting"}, {"from": "human", "value": "What is the percent disk_usage of disk path '/app' at 2025-02-12T03:00:00+07:00 from host '<vm-hostname>'"}, {"from": "gpt", "value": "At 2025-02-12T03:00:00+07:00, disk path '/app' from host '<vm-hostname>' has used 5.20% of total disk allocation: 5582209024 of 107317563392 bytes."}]}
```



### 6. Transform targeted datasets to language model template data (chatbot)

```python
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1"
)

def formatting_prompts_func(data):
    convos = data['conversations']
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts }

dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)
```

ตัวอย่าง output หลังจากแปลงข้อมูล

```json
print(dataset[0]['conversations'])

[
  {
    'content': 'You are an expert System Engineer, you will predict how much that our company has used CPU, Memory (GB) and Disk Usage (%) with our company data with really good assisting', 
    'role': 'system'
  }, 
  {
    'content': "What is the percent disk_usage of disk path '/app' at 2025-02-12T03:00:00+07:00 from host '<vm-hostname>'", 
    'role': 'user'
  }, 
  {
    'content': "At 2025-02-12T03:00:00+07:00, disk path '/app' from host '<vm-hostname>' has used 5.20% of total disk allocation: 5582209024 of 107317563392 bytes.", 
    'role': 'assistant'
  }
]
```

```python
print(dataset[0]['text'])

<｜begin▁of▁sentence｜><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

You are an expert System Engineer, you will predict how much that our company has used CPU, Memory (GB) and Disk Usage (%) with our company data with really good assisting<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the percent disk_usage of disk path '/app' at 2025-02-12T03:00:00+07:00 from host '<vm-hostname>'<|eot_id|><|start_header_id|>assistant<|end_header_id|>

At 2025-02-12T03:00:00+07:00, disk path '/app' from host '<vm-hostname>' has used 5.20% of total disk allocation: 5582209024 of 107317563392 bytes.<|eot_id|>
```



### 7. Begin training model on targeted datasets with **Supervised Fine-Tuning (SFT)**

```python
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
        output_dir = f"{model_dir}/outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
```

- **dataset_text_field:** กำหนดว่าใน `train_dataset` ฟิลด์ไหนที่เก็บข้อความที่ใช้ในการฝึก โดยปกติจะใช้ field "text" ตามตัวอย่าง output ข้อ 6
- **data_collator:** ใช้ในการเตรียมข้อมูลสำหรับการฝึกโมเดลที่มีลักษณะเป็น **Sequence-to-Sequence (Seq2Seq)** เช่น โมเดลที่ทำงานกับข้อมูลแบบคำถาม-คำตอบ, การแปลภาษา, หรือการสรุปข้อความ โดยหลักการแล้ว **data collator** จะทำหน้าที่ในการรวบรวมข้อมูลจากหลายๆ ตัวอย่างใน batch หนึ่ง เพื่อให้สามารถนำเข้าโมเดลได้อย่างถูกต้อง
- **TrainingArguments:**
  - **per_device_train_batch_size:** 
    - ขนาดของ batch_size ที่โมเดลใช้ฝึกข้อมูล (x samples/รอบ) 
    - ยิ่งกำหนดเยอะ ยิ่งเทรนได้จำนวนรอบเร็วขึ้น แต่ใช้ resource สูง 
    - เช่น มีข้อมูล 1 ล้าน row (ตามตัวอย่าง output ข้อ 6 ถือเป็น 1 row) หากกำหนดค่าเป็น 2 ดังนั้น โมเดลจะทำการเทรนจำนวน 5 แสนรอบ (รอบละ 2 samples) 
  - **gradient_accumulation_steps:** 
    - จำนวนขั้นที่จะสะสมค่า gradient ก่อนอัพเดทค่า weight 
    - โดยปกติจะทำการอัพเดทค่าทันที ซึ่งการอัพเดทค่าทุกรอบอาจทำให้การประมวลผลช้าลง ในกรณีที่ resource มีจำกัด เนื่องจากขนาดของ batch ที่ใหญ่ขึ้น หากสามารถชะลอการคำนวณและอัพเดทลงได้ จะช่วยทำให้ประหยัด resource ในการประมวลผลมากขึ้น
  - **weight_decay:** 
    - ค่าสำหรับปรับลดขนาด weight ให้เล็กลง เพื่อไม่ให้โมเดล Overfitting
    - ต่างจาก dropout ตรงที่ เป็นการปิดหรือลบช่องทางเรียนรู้โมเดลบางส่วนออกไปเลย แต่ decay จะเน้นไปที่การคำนวณผ่านค่า weight ตรง ๆ เพื่อให้ได้ค่า weight ที่เหมาะสมต่อการเรียนรู้โมเดล
  - **lr_scheduler_type:**
    - ประเภทของการปรับ learning rate โดยปกติจะปรับเป็นแบบ linear เพื่อให้ค่าลดลงอย่างค่อยเป็นค่อยไปในช่วงของการฝึก



### 8. Define responses-only data for Model training

โดยปกติแล้ว จะเน้น training แค่ส่วนที่ให้ chatbot ตอบเรามาเท่านั้น จึงจำเป็นต้องกำหนดว่า ส่วนใดของข้อมูล text เป็นส่วนของ input (เราเป็นคนถาม) และส่วนไหนเป็นส่วน output (chatbot เป็นคนตอบ)

ซึ่งหากข้อมูลถูกแปลงเป็น pattern ตามตัวอย่างข้อ 6 แล้ว จะสามารถแยกส่วนของ text ได้ โดยส่วนที่ต้อง config มี 2 ส่วนคือ

- **instruction_part:** ส่วนของ user (เข้าใจว่ากำหนดส่วนนี้เพื่อไม่ต้องนำไปเทรนเพิ่ม)
- response_part: ส่วนของ assistant (ส่วนที่จะถูก training แน่นอน เพื่อปรับปรุงรูปแบบการตอบคำถามแก่ user ให้มีความ smooth ยิ่งขึ้น)

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
```



### 9. Train Model

หากไม่ต้องการเก็บผลการเทรน เช่น ค่า loss, learning_rate, runtime, etc. สามารถลบการประกาศตัวแปรออกได้เลย

```python
trainer_stats = trainer.train()

# [Optiinal] Save Training Stat to file.
current_date = datetime.now().strftime("%Y%m%d-%H%M%S")
trainer_stats_filename = f"trainer-stats-{current_date}.txt"

with open(f"{model_dir}/trainer-stats/{trainer_stats_filename}", "w") as file:
    file.write(str(trainer_stats))

print(f"Trainer stats saved to {model_dir}/trainer-stats/{trainer_stats_filename}")
```



### 10. Save Model for Internal Usage or Container-Based Usage

```python
# no-container model saving.
model.save_pretrained(f"{model_dir}/pretrained-model")
tokenizer.save_pretrained(f"{model_dir}/pretrained-model")

# container-based model saving. GGUF usage
model.save_pretrained_gguf(f"{model_dir}/pretrained-model", tokenizer)
```

