# ----------------------------------------------------------
# Usage example:
# python scripts/run_embedding_extract.py \
#  --json data/opinionqa_500/opinionqa_option_strings.json \
#  --model meta-llama/Llama-2-7b-hf \
#  --out outputs/llm_embeddings \
#  --n_workers 1 \
#  --layer "all" \
#  --extract_position before_eos 
# ----------------------------------------------------------

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from sibyl.utils.llm_utils import get_embedding_fsdp
from sibyl.constants.string_registry_llm import MODEL_NAME_TO_NICKNAME
from sibyl.utils.logger import start_capture


SAVING_FORMAT = ("{dataset}_embedding_{model_nickname}"
                 +"{lora_name}_layer_{layer}_eos_{use_eos_position}.pt")
REFUSAL_INDICATOR = ["Refused", "Did not receive", "DK", "Unclear", "No answer"]

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text embeddings for input to the GNN."
    )
    parser.add_argument("--json", type=Path,
                        required=True,
                        help="Input JSON file.")
    parser.add_argument("--model", type=str,
                        required=True,
                        help="huggingface model name / path.")
    parser.add_argument("--lora_path", type=Path,
                        default=None,
                        help="(Optional) Lora module path to load.")
    parser.add_argument("--layer",
                        default="all",
                        help="Hidden-state layer to extract embeddings from. 'all' for all layers.")
    parser.add_argument("--n_workers", type=int,
                        default=1,
                        help="Number of FSDP workers.")
    parser.add_argument("--batch_size", type=int,
                        default=1,
                        help="Batch size per worker.")
    parser.add_argument("--extract_position", type=str,
                        choices=["eos", "before_eos"], default="before_eos",
                        help="Position to extract the embedding from.")
    parser.add_argument("--out", type=Path,
                        required=True,
                        help="Directory to save the dict (torch.save).")

    args = parser.parse_args()
    if args.layer == "all":
        _cfg = AutoConfig.from_pretrained(args.model)
        n_layers = _cfg.num_hidden_layers + 1

    print("Running with arguments: ", args)
    use_eos_position = True if args.extract_position == "eos" else False
    saving_format = SAVING_FORMAT.format(
        dataset=args.json.stem,
        model_nickname=MODEL_NAME_TO_NICKNAME[args.model],
        lora_name=f"_{args.lora_path.stem}" if args.lora_path else "",
        layer=str(args.layer),
        use_eos_position=use_eos_position
    )
    print(f"--> Saving saming filename: {saving_format}")

    with open(args.json, 'r') as f:
        key_text_dict = json.load(f)
        print(f"Loaded {len(key_text_dict)} samples from {args.json}")
    keys = list(key_text_dict.keys())
    texts = list(key_text_dict.values())

    embeddings : torch.Tensor = get_embedding_fsdp(
        model_name=args.model,
        text=texts,
        n_workers=args.n_workers,
        eos_position=use_eos_position,
        layer_idx=(
            args.layer if args.layer != "all" else list(range(n_layers))
        ),
        batch_size=args.batch_size,
        lora_path=args.lora_path,
    ) # eos_position=False to extract embeddings at ":" token
    assert embeddings.shape[0] == len(keys), (
        "--> embedding extraction: count mismatch."
    )
    key_to_emb = {k: v for k, v in zip(keys, embeddings)}

    save_filename = Path(args.out) / saving_format
    save_filename.parent.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {len(key_to_emb)} embeddings → {str(save_filename)}")
    torch.save(key_to_emb, save_filename)
    return


if __name__ == "__main__":
    main()
