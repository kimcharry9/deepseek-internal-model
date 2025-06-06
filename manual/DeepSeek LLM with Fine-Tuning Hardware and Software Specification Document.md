# DeepSeek: LLM with Fine-Tuning Hardware and Software Specification Document

[TOC]

## 1. Specification

### 1.1) Hardwares (Low Spec)

- CPU: 16
- Memory: 64 GB
- GPU: 16 GB (1 piece) **! Required !**
- Disk: 150GB
  - Huggingface: Dataset storage ~20GB++
  - Model storage ~50GB


### 1.2) OS Packages

- Ollama: 0.6.1

  ```
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- Huggingface: 0.29.3

- Python + Pip: 3.12.x

### 1.3) Python Libraries

- unsloth: 2025.3.14 -> (Free license http://github.com/unslothai/unsloth)

  ```shell
  pip3.12 install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
  ```

- transformers: 4.49.0

- torch: 2.6.0+cu124

- cuda: 7.5

- cuda toolkit: 12.4

- triton: 3.2.0

- xformers: 0.0.29.post3

- bitsandbytes: 0.45.3

- peft: 0.14.0

- tokenizers: 0.21.1

- datasets: 3.4.0

- accelerate: 1.5.2

- langchain: 0.3.21

### 1.4) [OPTIONAL] On-Cloud Provisioning (AWS-Based)

หากต้องการทดสอบเบื้องต้น สามารถสร้างเครื่อง EC2 สำหรับทดสอบโมเดลได้ ด้วยรายละเอียด AMI ด้านล่าง โดย AMI นี้จะ Support Library ที่ต้องใช้ทำงานเรียบร้อยแล้ว ลงเพิ่มเพียงแค่ Ollama และ Unsloth ตาม command แนบด้านบน

- **ami id:**
  - ami-0583d1c3289affd3c
- **ami name:**
  - Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Amazon Linux 2023) 20250302

- **ec2 types:**
  - g4dn.4xlarge (16/64) + 150GB Disk เป็นอย่างต่ำ
  - เป็น instance type ที่ราคาถูกที่สุด (1$/hour)  และใน region ที่ใช้งานอยู่มีแค่ type นี้ให้ใช้งาน (instance type ref อื่น ๆ: [AWS Instance Type](https://aws.amazon.com/ec2/instance-types/))