import streamlit as st

import os
import uuid
import json
import requests
import time 
from Table_RAG import TableRAG 


st.set_page_config(page_title="TableRAG")
st.title("Welcome to TableRAG")
st.write("Start by uploading a PDF contract")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

deployment_name = "gpt-4.1"

# --- Helper function to extract tables from PDF ---
def extract_tables(pdf_path, pdf_id, deployment_name):
    from html_to_jsonl_azure import get_client, process_html  # import only when needed

    data = {"doc_id": pdf_id, "accuracy": "high"}
    with open(pdf_path, "rb") as f:
        files = {"pdf_stream": f}
        response = requests.post("https://ail-dev.prgx.com/cciclause/table", data=data, files=files)

    clause_html, tables = None, []
    if response.status_code == 200:
        resp_json = response.json()
        categories = resp_json.get("categories", [])
        if categories and categories[0].get("clauses"):
            clause_html = categories[0]["clauses"][0].get("clause_text")
            if clause_html:
                client = get_client()
                tables, _ = process_html(client, deployment_name, clause_html, "clause_html", start_index=1)

                # Save tables to disk once
                for table in tables:
                    with open("/home/jovyan/git/TableRAG/data/raw/small_dataset/dummy.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(table, ensure_ascii=False) + "\n")
    return clause_html, tables

# --- PDF upload and extraction (runs once per file) ---
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file:
    pdf_name = uploaded_file.name

    if "pdf_data" not in st.session_state or st.session_state.pdf_data.get("name") != pdf_name:
        pdf_id = str(uuid.uuid4())
        pdf_path = os.path.join(UPLOAD_DIR, pdf_name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Running Table Extraction Service on uploaded PDF"):
            clause_html, tables = extract_tables(pdf_path, pdf_id, deployment_name)
            st.session_state.pdf_data = {
                "name": pdf_name,
                "clause_html": clause_html,
                "tables": tables
            }

# --- Use cached PDF extraction results ---
pdf_data = st.session_state.get("pdf_data", {})
clause_html = pdf_data.get("clause_html")
tables = pdf_data.get("tables", [])

if clause_html:
    st.success("Clause extracted successfully ✅")
    st.markdown(clause_html, unsafe_allow_html=True)

# --- User query input ---
if clause_html:
    user_query = st.text_input("Ask TableRAG a question about the table you submitted!")
    submit_query = st.button("Submit Query")

    if submit_query and user_query:
        start_time = time.time()
        with st.spinner("Running retrieval and generating response..."):
             
            tablerag = TableRAG()           # safe: only instantiated after extraction
            queries_path = "/home/jovyan/git/TableRAG/data/raw/small_dataset/queries.jsonl"
            file_path = "/home/jovyan/git/TableRAG/data/colbert_results/dummy_llm_based_filter_None_string_progress.json"
            query_obj = {"query": user_query, "label": ""}
            with open(queries_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(query_obj, ensure_ascii=False) + "\n")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("0")
            
            tablerag.run_colbert_retrieval()
            response = tablerag.generate_llm_response()
        elapsed = time.time() - start_time
        st.write(response)
        print(f"⏱ Elapsed time: {elapsed:.2f} seconds")