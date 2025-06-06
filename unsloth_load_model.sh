#########################################################################################
#                                                                                       #
#                    *** Machine Learning - Prediction Model ***                        #
#             	    DeepSeek LLM Training via Unsloth & Ollama POC               	#
#                    Implemented by Placid SE-AI Development Team                       #
#                                                                                       #
#########################################################################################

#!/bin/bash

path="/data/deepseek-model"
model_name="DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
export HF_HOME="${path}/${model_name}/huggingface"
export HF_DATASETS_CACHE="${path}/${model_name}/huggingface/datasets"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_TIME=$(date +%s)

# Remove Cache before executing.
rm -rf ${path}/${model_name}/huggingface

source ${path}/venv/bin/activate
python3.12 ${path}/${model_name}/train-program/unsloth_load_model.py

END_TIME=$(date +%s)
TOTAL_RUNNING_TIME=$((END_TIME-START_TIME))
echo "[`date +"%Y-%m-%d %H:%M:%S"`] Total time: ${TOTAL_RUNNING_TIME} s"
