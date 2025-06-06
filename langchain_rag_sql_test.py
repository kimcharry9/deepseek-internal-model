import json
import pandas as pd
import glob
import time
import torch
import sys
from tqdm import tqdm
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain

model_dir = "/data/deepseek-model/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
data_dir = "/data/deepseek-model/cleansed_dataset"
#model = "deepseek-r1-8b-param-ver20250321_145325"
model = "deepseek-r1:14b"
sql_db = "/data/sqlite_db/disk_usage.db"
embedding_model = HuggingFaceEmbeddings()

#input_host = sys.argv[1]
#input_date = sys.argv[2]
#input_time = sys.argv[3]
#input_path = sys.argv[4]

######################################################### 
#   RAG Processing                                      #
#########################################################

#-- Load Datasets from SQLite DB --#
def load_model():
    print("[1] Load Datasets from SQLite DB: Processing...")
    db = SQLDatabase.from_uri(f"sqlite:///{sql_db}")
    print("[1] Load Datasets from SQLite DB: Success!")

    # Generate answers
    #llm_chain = LLMChain(llm=Ollama(model=model), prompt=prompt)
    print("[2] Configure LLM Chain: Processing...")
    llm = OllamaLLM(model=model)
    print("[2] Configure LLM Chain: Success!")

    return db, llm
    
#query = f"What is the percent of disk_usage for '{input_host}' on '{input_date}' at '{input_time}' for path '{input_path}'?"
#query = f"Please summarize the insight of disk_usage for '{input_host}' for path '{input_path}' from database and which date and time have the most disk_usage?"

def build_sql_query(user_query, input_host, input_path, input_date, input_time):
    if "summarize" in user_query or "summary" in user_query:
        sql_query = f"SELECT date, AVG(disk_used_bytes), AVG(disk_used_percent) FROM disk_usage WHERE hostname LIKE '%{input_host}' AND disk_path LIKE '%{input_path}' GROUP BY date;"
    #elif "date" in user_query:
    else:
        sql_query = f"SELECT date, time, disk_used_bytes, disk_total, disk_used_percent FROM disk_usage WHERE hostname LIKE '%{input_host}' AND disk_path LIKE '%{input_path}' AND date = '{input_date}' AND time = '{input_time}';"
    return sql_query

def chat_loop(db, llm):
    print("DeepSeek-R1 Model Greetings! Please feel free to ask your information here.")

    while True:
        user_query = input("Q: ").strip()
        if user_query.lower() == "exit" or user_query.lower() == "":
            break
        if "disk" in user_query.lower():
            input_host = input("Please type your targeted hostname: ").strip()
            input_path = input("Please type your targeted path: ").strip()
            if "on date" in user_query.lower():
                input_date = input("Please type your targeted date: ").strip()
                input_time = input("Please type your targeted time: ").strip()
            else:
                input_date = ""
                input_time = ""
            sql_query = build_sql_query(user_query.lower(), input_host, input_path, input_date, input_time)
            sql_result = db.run(sql_query)
        else:
            sql_query = build_sql_query("", "", "", "", "")
            sql_result = db.run(sql_query)

        # Craft the prompt template
        print("[3] Pull Prompt Template and Retrieve SQL Query: Processing...")
        formatted_results = f"The query returned the following result: [(date, time, disk_used_bytes, disk_total, disk_used_percent)] is {json.dumps(sql_result, indent=2)}"
        print(formatted_results)
        prompt_template = PromptTemplate(
            input_variables=["user_query", "formatted_results"],
            template=
                f"""
                You are a system monitoring assistant. Use the following SQL result to answer the user's question in a clear and concise way.

                The user asked:
                {user_query}

                The query result:
                {formatted_results}

                Your answer should explain:
                - Analyze data with the user asked requirement "{user_query}" and sql_query "{sql_query}" in SQLite Databases, Only use the columns: date, time, hostname, disk_path, disk_used_bytes, disk_total, disk_used_percent. Make sure to filter by date or time if the user asks about a specific time period.
                - Please answer with exactly value in databases.
                - Please keep the previous user asked requirement to continue answering the similar details.
                """
        )
        print("[3] Pull Prompt Template and Retrieve SQL Query: Success!")

        print("[4] Generate answer with deepseek-model: Processing...")
        #retriever = create_sql_agent(llm_chain, db=db, agent_type="zero-shot-react-description", verbose=True)
        formatted_prompt = prompt_template.format(user_query=user_query, formatted_results=formatted_results)
        retriever = llm.invoke(formatted_prompt)
        print(f"\n A: {retriever}\n{'-'*60}\n")
        print("[4] Generate answer with deepseek-model: Success!")

if __name__ == "__main__":
    db, llm = load_model()
    chat_loop(db, llm)
