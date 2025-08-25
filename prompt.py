from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
import typing
import os
import json

# Load environment variables
load_dotenv()

# Azure OpenAI client
azure = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

# Path to ColBERT retrieval results
retrieval_results_path = Path(
    "/home/jovyan/git/TableRAG/data/colbert_results/dummy_llm_based_filter_None_string_retrieval_results.jsonl"
)


def azure_ai_generate_llm_response(
    prompt: str,
    system_prompt: str = (
        "You will be given a query to be answered, and the results of ColBERT retrieval on a group of tables.\n"
        "Find the appropriate answer to the user's query and respond. Your answer must be based on the information in the table."
    ),
    model: typing.Literal["gpt-4o-mini", "gpt-4.1"] = "gpt-4o-mini",
) -> str:
    """
    Simple wrapper around Azure OpenAI chat completions API.

    Args:
        prompt (str): User query or prompt.
        system_prompt (str): System instruction for the assistant.
        model (str): Model to use (e.g., "gpt-4o-mini" or "gpt-4.1").

    Returns:
        str: LLM-generated response.
    """
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


def generate_responses_from_colbert_results(file_path: Path, model: str = "gpt-4.1"):
    """
    Iterate through ColBERT retrieval results and generate answers using Azure OpenAI.

    Args:
        file_path (Path): Path to the JSONL file containing ColBERT retrieval results.
        model (str): Azure OpenAI model to use.
    """
    with file_path.open("r") as f:
        for line in f:
            data = json.loads(line)
            query = data.get("query", "")
            retrieved_docs = data.get("retrieved_docs", [])
            table_context = "\n\n".join(doc.get("table_formatted", "") for doc in retrieved_docs)

            prompt = f"Question: {query}\n\nTable:\n{table_context}\n\nAnswer:"
            answer = azure_ai_generate_llm_response(prompt, model=model)

            print(f"Query: {query}")
            print(f"Answer: {answer}")
            print("=" * 50)


# Example usage
if __name__ == "__main__":
    generate_responses_from_colbert_results(retrieval_results_path, model="gpt-4.1")
