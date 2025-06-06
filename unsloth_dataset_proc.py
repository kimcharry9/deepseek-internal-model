from datasets import load_dataset
from unsloth import to_sharegpt, standardize_sharegpt

dataset = load_dataset("json", data_files="/data/deepseek-model/cleansed_dataset/system_disk_usage_json2_500k_trim.jsonl", split="train")
print(dataset.column_names)
dataset[0]

dataset = to_sharegpt(
    dataset,
    merged_prompt = "{instruction}[[\nYour input is:\n{input}]]",
    output_column_name = "output",
    conversation_extension = 3, # Select more to handle longer conversations
)

dataset = standardize_sharegpt(dataset)
dataset[0]['conversations']
