import argparse
import json
import os
from pathlib import Path

import torch
from openai import OpenAI
from tqdm import tqdm

MAX_BATCH = 100

def run(filepath: Path, model_name:str) -> None:
    with open(filepath, "r") as f:
        data = json.load(f)

    # prepare a dictionary which key = question keys, value = question texts
    qkey_to_text = {}
    for key, item in data.items():
        qkey = key.split("_option_")[0]
        text = item.split("Answer:")[0] + "Answer:"
        if qkey in qkey_to_text:
            assert qkey_to_text[qkey] == text
        qkey_to_text[qkey] = text
    print(f"Total number of questions: {len(qkey_to_text)}")

    # if-else client instance dependent on the text embedding model.
    # change according to text embedding model of your choice.
    if "gemini" in model_name:
        client = OpenAI(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif "text-embedding" in model_name:
        client = OpenAI()

    # API call for text embeddings.
    texts = list(qkey_to_text.values())
    text_embeddings = []
    for i in tqdm(range(0, len(texts), MAX_BATCH), desc="Generating embeddings batch"):
        bs = min(MAX_BATCH, len(texts) - i)
        batch_texts = texts[i : i + bs]
        response = client.embeddings.create(
            input=batch_texts, model=model_name,
        )
        batch_embeddings = [
            torch.tensor(d.embedding, dtype=torch.float32) for d in response.data
        ]
        text_embeddings.extend(batch_embeddings)

    llm_embeddings = {}
    qkeys = list(qkey_to_text.keys())
    for i in range(len(texts)):
        llm_embeddings[qkeys[i]] = text_embeddings[i]

    dataset_name = filepath.parent.name
    save_filepath = f"outputs/text_embeddings/{dataset_name}_text_embeddings_{model_name}.pth"
    os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
    torch.save(llm_embeddings, save_filepath)
    print(f"Saved embeddings to: {save_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Get text embeddings.")
    parser.add_argument("--json", type=Path,
                        required=True,
                        help="Input JSON file.")
    parser.add_argument("--model_name", type=str,
                        default="gemini-embedding-001",
                        help="API text embedding model.")
    args = parser.parse_args()
    run(filepath=args.json, model_name=args.model_name)


if __name__ == "__main__":
    main()
