import json
import pandas as pd
import glob
import time
import torch
import sys
from tqdm import tqdm
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

model_dir = "/data/deepseek-model/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
data_dir = "/data/deepseek-model/cleansed_dataset"
db_dir = "/data/deepseek-model/faiss_db"
#model = "deepseek-r1-8b-param-ver20250321_145325"
model = "deepseek-r1:32b"
embedding_model = HuggingFaceEmbeddings()

input_host = sys.argv[1]
input_date = sys.argv[2]
input_time = sys.argv[3]
input_path = sys.argv[4]

######################################################### 
#   RAG Processing                                      #
#########################################################

#-- Load Datasets from FAISS DB --#
print("[1] Load Datasets from FAISS DB: Processing...")
db = FAISS.load_local(db_dir, embedding_model, allow_dangerous_deserialization=True)
print("[1] Load Datasets from FAISS DB: Success!")

query = f"What is the percent of disk_usage for '{input_host}' on '{input_date}' at '{input_time}' for path '{input_path}'?"
filters = {
    "hostname": f"{input_host}",
    "date": f"{input_date}",
    "time": f"{input_time}",
    "disk_path": f"{input_path}",
}

print("[2] Retrieve Data with k=2: Processing...")
retriever = db.as_retriever(search_kwargs={"k": 2})
print("[2] Retrieve Data with k=2: Success!")
#output = retriever.invoke(query, filter=filters)
#print(output)

# Craft the prompt template
print("[3] Pull Prompt Template: Processing...")
prompt = hub.pull("rlm/rag-prompt")
print("[3] Pull Prompt Template: Success!")

# Chain 1: Generate answers
#llm_chain = LLMChain(llm=Ollama(model=model), prompt=prompt)
print("[4] Configure LLM Chain: Processing...")
llm_chain = OllamaLLM(model=model)
print("[4] Configure LLM Chain: Success!")

# Chain 2: Combine document chunks
def format_docs(db):
    return "\n\n".join(doc.page_content for doc in db)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_chain
    | StrOutputParser()
)

#result = rag_chain.invoke(f"What is the percent of disk_usage for '{input_host}' on '{input_date}' at '{input_time}' for path '{input_path}'?")
print("[5] Generate answer with deepseek-model: Processing...")
result = rag_chain.invoke(query, filter=filters)
print(result)
print("[5] Generate answer with deepseek-model: Success!")
