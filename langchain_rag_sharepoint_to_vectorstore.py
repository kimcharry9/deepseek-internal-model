import json
import pandas as pd
import glob
import time
import torch
import sys
import os
import shutil
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
from langchain_community.document_loaders.sharepoint import SharePointLoader
from langchain_community.document_loaders.parsers.msword import MsWordParser
from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
from dotenv import load_dotenv
from pprint import pprint

# Load credentials from .env
load_dotenv(override=True)

# Define destination path for sharepoint file collection
#output_folder = "/data/open-webui_data/sync_dir"
model_dir = "/data/deepseek-model/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
data_dir = "/data/deepseek-model/cleansed_dataset"
#model = Ollama(model="deepseek-r1-8b-param-ver20250321_145325")
#model = "deepseek-r1-8b-param-ver20250321_145325"
model = "deepseek-r1:32b"
embedding_model = HuggingFaceEmbeddings()
vectorstore_dir = "/data/deepseek-model/faiss_db" 
os.makedirs(vectorstore_dir, exist_ok=True)

# Load document_library_id from .env
document_library_id = os.getenv("DOCUMENT_LIBRARY_ID")

# `handlers` keys must be either file extensions or mimetypes. 
# using file extensions:
#handlers = {
#    "doc": MsWordParser(),
#    "pdf": PDFMinerParser(),
#    "mp3": OpenAIWhisperParser()    
#}

# using MIME types:
handlers = {
    "application/msword": MsWordParser(),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": MsWordParser(),
    "application/pdf": PDFMinerParser(),
    "audio/mpeg": OpenAIWhisperParser(),
    "image/jpeg": OpenAIWhisperParser()
}

# Load sharepoint's files
doc_loader = SharePointLoader(
    document_library_id=document_library_id,
    folder_path="read_doc_test",
    recursive=True,
    auth_with_token=True,
    handlers=handlers
)

documents = doc_loader.load()

# ดูโครงสร้างของเอกสารแรก
#pprint(documents[0].__dict__)

print("[1] Create UUID: Processing...")
with tqdm(range(len(documents)), desc="Generate uuid to documents...", unit="pieces") as doc_ids:
    uuids = [str(uuid4()) for _ in doc_ids]
print("[1] Create UUID: Completed!")

print("[2] Ingest Document in vectorstore format: Processing...")
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
#documents = text_splitter.split_documents(documents)

with tqdm(total=len(documents), desc="Ingesting documents to vectorstore", unit="docs") as vec:
    db.add_documents(documents=documents, ids=uuids)
    vec.update(len(documents))
    #db = FAISS.from_documents(documents, embedding_model)
print("[2] Ingest Document in vectorstore format: Completed!")

print("[3] Save vectorstore to local disk: Processing...")
db.save_local(vectorstore_dir)
print("[3] Save vectorstore to local disk: Completed!")
