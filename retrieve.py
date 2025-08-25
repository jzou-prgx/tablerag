from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
import typing
import os
 
load_dotenv()
 
retrieval_results_path = "/home/jovyan/git/TableRAG/data/colbert_results/dummy_llm_based_filter_None_string_retrieval_results.jsonl"
azure = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)
 
 
def azure_ai_generate_llm_response(
    prompt: str,
    system_prompt: str = (
    "You will be given a query to be answered, and the results of colBERT retrieval on a group of tables.\n"
    "Find the appropriate answer to the user's query and respond. Your answer must be based on the information in the table."
    )

    model: typing.Literal["gpt-4o-mini", "gpt-4.1"] = "gpt-4o-mini",
):
    """
    Simple wrapper of Azure AI chat.completions API to make calling an LLM into
    a str -> str like function.
 
    Args:
        prompt (str): User chat message
        system_prompt (str): System chat message (top of chat)
        model (str): Which model to use, like "gpt-4o-mini"
 
    Returns:
        str: LLM response
    """
    response = azure.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_completion_tokens=2_000,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=model,
    )
    return response.choices[0].message.content

    def generate_response():
