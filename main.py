
from src.table_processor.table_processing_pipeline import table_processing_pipeline

table_processing_pipeline(
        task_name = "dummy",
        split="train",
        table_filter_name= "llm_based_filter",
        table_clarifier_name= "None",#term_explanations_and_table_summary",
        embedding_type="text-embedding-3-large",
        top_k=5,
        save_jsonl=True,
        load_local_dataset=True,
        experiment_name="custom_table_clarification",
        use_sampled_table_for_augmentation=False,
        sample_size=15,
        overwrite_existing=True,
        table_format="string"
)
print("we got here")
'''
CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_pipeline.retriever \
  --dataset_path /home/jovyan/git/TableRAG/data/processed/dummy_llm_based_filter_None_string.jsonl \
  --index_name dummy_index \
  --colbert_model_name colbert-ir/colbertv2.0 \
  --base_output_dir /home/jovyan/git/TableRAG/data/colbert_results \
  --query_grd_path /home/jovyan/git/TableRAG/data/raw/small_dataset/dummy.jsonl \
  --top_k 1 \
  --rerank_top_k 1 \
  --num_queries 15
  '''