from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json
import typing

# Load environment variables
load_dotenv()

# Azure OpenAI client
azure = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

# Paths
queries_path = Path("/home/jovyan/git/TableRAG/data/raw/small_dataset/queries.jsonl")  # JSONL with queries
retrieval_results_path = Path("/home/jovyan/git/TableRAG/data/colbert_results/dummy_llm_based_filter_None_string_retrieval_results.jsonl")

def azure_ai_generate_llm_response(
    prompt: str,
    system_prompt: str = (
        "You will be given a query to be answered, and the results of ColBERT retrieval on a group of tables.\n"
        "Find the appropriate answer to the user's query and respond. Your answer must be based on the information in the tables. Do not hallucinate answers or try to come up with an answer through conjecture"
    ),
    model: typing.Literal["gpt-4o-mini", "gpt-4.1"] = "gpt-4o-mini",
) -> str:
    response = azure.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=2000,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=model,
    )
    return response.choices[0].message.content


def generate_responses_from_queries(queries_file: Path, colbert_file: Path, model: str = "gpt-4.1"):
    # Load ColBERT retrieval results into a dictionary keyed by query
    colbert_dict = {}
    with colbert_file.open("r") as f:
        for line in f:
            data = json.loads(line)
            query_text = data.get("query", "")
            colbert_dict[query_text] = data.get("retrieved_docs", [])

    # Iterate over queries
    with queries_file.open("r") as f:
        for line in f:
            qdata = json.loads(line)
            query = qdata.get("query", "")
            retrieved_docs = colbert_dict.get(query, [])
            table_context = "\n\n".join(doc.get("table_formatted", "") for doc in retrieved_docs)

            prompt = f"Question: {query}\n\nTable:\n{table_context}\n\nAnswer:"
            answer = azure_ai_generate_llm_response(prompt, model=model)

            print(f"Query: {query}")
            print(f"Answer: {answer}")
            print("=" * 50)
    return answer


if __name__ == "__main__":
    answers = generate_responses_from_queries(queries_path, retrieval_results_path, model="gpt-4.1")
