import pandas as pd
import datetime
import glob
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

## 1. Data Ingestion & Cleansing

path = "/data/deepseek-model"
db_dir = f"{path}/faiss_db"
date_range = [str(i) for i in range(20250218, 20250219)]
    
file_paths = []
for date in date_range:
    file_paths.extend(glob.glob(f"{path}/mockdata/{date}/*gsb-lake-prd-monitor02_8086_telegraf_disk.csv.gz"))

df_list = [pd.read_csv(file, compression='gzip') for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

# clean missing values
df = df.dropna()

# แปลง timestamp เป็น datetime format
df["time"] = df["time"].astype(str)

print(df)

## 2. Data Transformation - from csv. to text (message) data

datasets = []

for _, row in df.iterrows():
    timestamp = row["time"]
    disk_path = row["path"]
    hostname = row["hostname"]
    disk_used_percent = float(row["disk_used.percent"])
    disk_total = int(row["disk_total.bytes"])
    disk_used_bytes = int(row["disk_used.bytes"])

    instruction_text = f"You are an expert System Engineer, you will predict how much that our company has used CPU, Memory (GB) and Disk Usage (%) with our company data with really good assisting"
    human_text = f"What is the percent disk_usage of disk path '{disk_path}' at {timestamp} from host '{hostname}'"
    gpt_text = f"At {timestamp}, disk path '{disk_path}' from host '{hostname}' has used {disk_used_percent:.2f}% of total disk allocation: {disk_used_bytes} of {disk_total} bytes."

    tmp_doc = {
        "instruction": instruction_text,
        "query": human_text,
        "response": gpt_text,
        "metadata": row.to_dict()
    }
    datasets.append(tmp_doc)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts([d["response"] for d in datasets], embeddings)
vector_db.save_local(db_dir)
