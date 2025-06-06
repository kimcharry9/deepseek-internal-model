import json
import pandas as pd
import glob
import time
import torch
import os
import sys
import faiss
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

model_dir = "/data/deepseek-model/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
data_dir = "/data/deepseek-model/cleansed_dataset"
db_dir = "/data/deepseek-model/faiss_db"
#model = Ollama(model="deepseek-r1-8b-param-ver20250321_145325")
#model = "deepseek-r1-8b-param-ver20250321_145325"
model = "deepseek-r1:32b"
embedding_model = HuggingFaceEmbeddings()

######################################################### 
#   RAG Processing                                      #
#########################################################

#-- Load Datasets to Document Format --#
date_range = [str(i) for i in range(20250218, 20250219)]

file_paths = []
documents = []
for date in date_range:
    file_paths.extend(glob.glob(f"/data/deepseek-model/mockdata/{date}/*gsb-lake-prd-monitor02_8086_telegraf_disk.csv.gz"))

print("[1] Pull and Generate Datasets: Processing...")
with tqdm(file_paths, desc=f"Read CSV File {file_paths}...", unit="files") as raw_data:
    for file in raw_data:
        df = pd.read_csv(file, compression='gzip').dropna()
        df["time"] = pd.to_datetime(df['time'])  # แปลง timestamp

        with tqdm(df.iterrows(), total=len(df), desc="- Ingesting documents...", unit="rows") as datasets:
            for _, row in datasets:
                doc = Document(
                    page_content=f"Percent of disk_usage for '{row['hostname']}' on '{row['time'].date().isoformat()}' at '{row['time'].time().isoformat()}' for path '{row['path']}' has used {row['disk_used.percent']:.2f}% of total disk allocation: {int(row['disk_used.bytes'])} of {int(row['disk_total.bytes'])} bytes.",
                    metadata={
                        "date": row["time"].date().isoformat(),
                        "time": row["time"].time().isoformat(),
                        "hostname": row["hostname"],
                        "disk_path": row["path"],
                        "disk_used_percent": float(row["disk_used.percent"]),
                        "disk_total": int(row["disk_total.bytes"]),
                        "disk_used_bytes": int(row["disk_used.bytes"])
                    }
                )
                documents.append(doc)
print("[1] Pull and Generate Datasets: Completed!")

print("[2] Create UUID: Processing...")
with tqdm(range(len(documents)), desc="Generate uuid to documents...", unit="pieces") as doc_ids:
    uuids = [str(uuid4()) for _ in doc_ids]
print("[2] Create UUID: Completed!")

print("[3] Ingest Document in vectorstore format: Processing...")

index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))

db = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

#-- Generate Datasets via Chroma into vectorstore --#
#text_splitter = RecursiveCharacterTextSplitter(
#    chunk_size=1000,
#    chunk_overlap=200
#)
#splits = text_splitter.split_documents(documents)

with tqdm(total=len(documents), desc="Ingesting documents to vectorstore", unit="docs") as vec:
    db.add_documents(documents=documents, ids=uuids)
    vec.update(len(documents))
    #db = FAISS.from_documents(documents, embedding_model)
print("[3] Ingest Document in vectorstore format: Completed!")

print("[4] Save vectorstore to local disk: Processing...")
db.save_local(db_dir)
print("[4] Save vectorstore to local disk: Completed!")
