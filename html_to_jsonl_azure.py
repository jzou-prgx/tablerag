#!/usr/bin/env python3
# html_to_jsonl_azure.py

import os
import json
import argparse
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()  # loads .env automatically if present

def get_client():
    """Initialize AzureOpenAI client from environment variables."""
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

PROMPT = """You are an expert at parsing HTML tables.

Task:
- Input: An HTML document containing one or more tables (possibly nested or irregular). A file may contain multiple tables.
- Output: A strict JSON object for each table with this format:

{
  "id": "products_1",
  "table": {
    "header": ["Product", "Price", "Stock"],
    "rows": [
      ["Apple", "1.00", "100"],
      ["Banana", "0.50", "150"],
      ["Orange", "0.75", "200"]
    ],
    "caption": "Inventory of products"
  },
  "query": "",
  "label": ""
}

Rules:
1. Each JSON object represents **one rectangular table**.
2. Include all rows, even subtotal or summary rows.
3. If a table is nested inside another, flatten it into a **separate JSON object** and replace the parent cell with a placeholder token like "[SEE_products_2]".
4. Ensure the table is rectangular: pad missing cells with empty strings "".
5. Always include "caption" if present; otherwise use "".
6. If multiple tables exist in the file, generate multiple JSON objects, one per table.
7. IDs must be based on the HTML filename, plus a counter suffix (_1, _2, ...).
8. Return ONLY valid JSON (one object per table). No explanations, no prose.

"""

import re

def process_html(client, model, html_content: str, file_stem: str, start_index: int):
    """Send HTML to GPT and return list of JSON objects with filename-based IDs."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": html_content}
        ]
    )
    content = resp.choices[0].message.content.strip()

    # Split multiple JSON objects using regex (assumes objects start with '{' and end with '}')
    json_objects = re.findall(r'\{.*?\}(?=\s*\{|\s*$)', content, flags=re.DOTALL)

    objs = []
    idx = start_index
    for obj_str in json_objects:
        parsed = json.loads(obj_str)
        if isinstance(parsed, dict):
            parsed["id"] = f"{file_stem}_{idx}"
            idx += 1
            objs.append(parsed)

    return objs, idx

def main():
    parser = argparse.ArgumentParser(description="Convert HTML tables to JSONL using Azure OpenAI GPT-4.1")
    parser.add_argument("--in_dir", required=True, help="Folder containing HTML files")
    parser.add_argument("--out_file", required=True, help="Output JSONL file")
    parser.add_argument("--deployment_name", required=True, help="Azure OpenAI deployment name for GPT-4.1")
    args = parser.parse_args()

    client = get_client()
    all_objects = []

    html_files = sorted(list(Path(args.in_dir).glob("*.html")))
    #html_files = html_files[:10]
    total_files = len(html_files)

    for i, html_file in enumerate(html_files, start=1):
        file_stem = html_file.stem
        html = html_file.read_text(encoding="utf-8", errors="ignore")
        print(f"Processing file {i}/{total_files}: {html_file.name} ...")
        objs, _ = process_html(client, args.deployment_name, html, file_stem, 1)
        all_objects.extend(objs)
        print(f"  ✅ Extracted {len(objs)} table(s)")

    with open(args.out_file, "w", encoding="utf-8") as f:
        for obj in all_objects:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"\n✅ Done! Wrote {len(all_objects)} tables to {args.out_file}")

if __name__ == "__main__":
    main()
