import json
import pandas as pd
import glob
import time
import torch
import streamlit as st
from tqdm import tqdm
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

model_dir = "/data/deepseek-model/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
data_dir = "/data/deepseek-model/cleansed_dataset"
#model = "deepseek-r1-8b-param-ver20250321_145325"
model = "deepseek-r1:32b"
#sql_db = "/data/sqlite_db/disk_usage.db"
db_dir = "/data/deepseek-model/faiss_db"

embedding_model = HuggingFaceEmbeddings()

######################################################### 
#   RAG Processing                                      #
#########################################################

#-- Load Datasets from FAISS DB --#
print("[1] Load Datasets from FAISS DB: Processing...")
db = FAISS.load_local(db_dir, embedding_model, allow_dangerous_deserialization=True)
print("[1] Load Datasets from FAISS DB: Success!")

print("[2] Retrieve Data with k=2: Processing...")
retriever = db.as_retriever(search_kwargs={"k": 2})
print("[2] Retrieve Data with k=2: Success!")

# Generate answers
#llm_chain = LLMChain(llm=Ollama(model=model), prompt=prompt)
print("[3] Configure LLM Chain: Processing...")
llm = OllamaLLM(model=model)
print("[3] Configure LLM Chain: Success!")

# Craft the prompt template
print("[4] Pull Prompt Template and Retrieve SQL Query: Processing...")

prompt_template = PromptTemplate(
input_variables=["user_query", "formatted_results"],
    template=
        f"""
        You are a Microsoft Office 365 Expert. Use the user asked to answer or find some detail from vectorstore.

        The user asked:
        {user_query}

        The query result:
        {retriever}
        
        Your answer should explain what is the detail of files or recap about those files that be imported from sharepoint (office 365).
        """
)
print("[4] Pull Prompt Template and Retrieve SQL Query: Success!")

print("[5] Run Streamlit UI with DeepSeek Model Deployment: Processing...")
st.set_page_config(page_title="💬 LLM Chat with FAISS", layout="wide")
st.title("💬 Ask your documents (RAG via Ollama)")

query = st.text_input("Ask a question about your data:")

formatted_prompt = prompt_template.format(user_query=query, formatted_results=retriever)

if query:
    with st.spinner("Thinking..."):
        response = llm.invoke(formatted_prompt)
        st.success("✅ Answer:")
        st.write(response)
print("[5] Run Streamlit UI with DeepSeek Model Deployment: Success!")
