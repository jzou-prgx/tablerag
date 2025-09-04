import subprocess
import os
import sys
from src.table_processor.table_processing_pipeline import table_processing_pipeline
from src.colbert_pipeline import retriever
from prompt import generate_responses_from_queries
from pathlib import Path
#TODO Query needs to be taken in as a user argument, not a file. 
#TODO Need to figure out how to get past the progress trackers implemented by TableRAG
#TODO Doing manual file deletion atm, need to kill that
#TODO Have some way of allowing additions to be made to the dataset. 

class TableRAG:
    def __init__(self):
        #self.file_path = file_path
        #self._validate_file()
        #self.run_html_json_converter()
        self.run_table_chunker()
    
    # def _validate_file(self):
    #    if not os.path.isfile(self.file_path):
     #       raise FileNotFoundError(f"File not found: {self.file_path}")
    def _run_command(self, cmd):
        subprocess.run(cmd, check=True, env=os.environ)
    def run_html_json_converter(self):
        cmd = [
            "python3",
            "html_to_jsonl_azure.py",
            "--in_dir", "/home/jovyan/git/TableRAG/azure_docl_ground_truth",
            "--out_file", "/home/jovyan/git/TableRAG/tables.jsonl",
            "--deployment_name", "gpt-4.1"
        ]
        self._run_command(cmd)

    def run_table_chunker(self):
        jsonl_path = "/home/jovyan/git/TableRAG/data/raw/small_dataset/dummy.jsonl"

        # Count the number of lines (tables) in the file
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                num_rows = sum(1 for _ in f)
        table_processing_pipeline(
            task_name="dummy",
            split="train",
            table_filter_name="llm_based_filter",
            table_clarifier_name="None", 
            embedding_type="text-embedding-3-large",
            top_k=5,
            save_jsonl=True,
            load_local_dataset=True,
            experiment_name="custom_table_clarification",
            use_sampled_table_for_augmentation=False,
            sample_size=num_rows,  # dynamically set
            overwrite_existing=True,
            table_format="string"
        )
    
    


    def run_colbert_retrieval(self):

        retriever.generate_retrieval_results(
        dataset_path="/home/jovyan/git/TableRAG/data/processed/dummy_llm_based_filter_None_string.jsonl",
        index_name="dummy_index",
        colbert_model_name="colbert-ir/colbertv2.0",
        base_output_dir="/home/jovyan/git/TableRAG/data/colbert_results",
        use_rerank=False,  
        top_k=3,
        rerank_top_k=1,
        num_queries=1,
        query_grd_path="/home/jovyan/git/TableRAG/data/raw/small_dataset/queries.jsonl"
        )

    def generate_llm_response(self):

        queries_path = Path("/home/jovyan/git/TableRAG/data/raw/small_dataset/queries.jsonl")  # JSONL with queries
        retrieval_results_path = Path("/home/jovyan/git/TableRAG/data/colbert_results/dummy_llm_based_filter_None_string_retrieval_results.jsonl")
        answer = generate_responses_from_queries(queries_path, retrieval_results_path, "gpt-4.1")
        return answer