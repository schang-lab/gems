import argparse
import datetime
import json
import os
import re
import pathlib
from typing import List, Tuple, Dict

import numpy as np
from vllm.lora.request import LoRARequest

from sibyl.utils.logger import start_capture
from sibyl.utils.llm_utils import get_llm_engine, cli_args_parser
from sibyl.utils.string_utils import prompt_chat_formatter

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]


def inference_offline(args, data_list, sampling_params, llm):
    """
    Offline batched inference for input_prompts in the data_list.
    As mentioned in Appendix, gpt-oss-20b and qwen-3 models require special token handling.
    """
    tokenizer = llm.get_tokenizer()
    # prepare the alphabet (A, B, ...) tokens for logprob extraction
    alphabet_coded: List[Tuple[int, int]] = [
        tuple([
            tokenizer.encode(" " + chr(ord("A") + idx), add_special_tokens=False)[-1],
            tokenizer.encode(chr(ord("A") + idx), add_special_tokens=False)[-1],
        ]) for idx in range(26) # A-Z
    ]

    # run inference
    prompts, targets = [data['prompt'] for data in data_list], [data['label'] for data in data_list]
    if args.is_chat:
        prompts = [prompt_chat_formatter(prompt) for prompt in prompts]
    lora_request = LoRARequest(
        args.lora_name, 1, lora_path=args.lora_path
    ) if args.lora_path is not None else None

    if llm.llm_engine.model_config.model == "openai/gpt-oss-20b":
        prompts = [
            tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False
            ) + "<|channel|>final<|message|>" for prompt in prompts
        ]
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    elif llm.llm_engine.model_config.model.startswith("Qwen/Qwen3"):
        prompts = [
            tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ) for prompt in prompts
        ]
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = (
            llm.chat(prompts, sampling_params, lora_request=lora_request)
            if args.is_chat
            else llm.generate(prompts, sampling_params, lora_request=lora_request)
        )                
    del llm

    # process the outputs
    results = []
    sum_probs = []
    total_samples, correct_samples = len(data_list), 0
    for idx, output in enumerate(outputs):
        logprobs = output.outputs[0].logprobs[0]
        len_options = 8
        prob_per_option = []
        for opt_idx in range(len_options):
            token_wos, token_ws = alphabet_coded[opt_idx][0], alphabet_coded[opt_idx][1] # w/o and w/ prefix space
            logprob_wos, logprob_ws = logprobs.get(token_wos, None), logprobs.get(token_ws, None)
            prob_1 = np.exp(logprob_wos.logprob) if logprob_wos is not None else 0
            prob_2 = np.exp(logprob_ws.logprob) if logprob_ws is not None else 0
            prob_per_option.append((prob_1 + prob_2) / (2 if token_wos == token_ws else 1))
        max_idx = np.argmax(prob_per_option)
        sum_probs.append(sum(prob_per_option))
        is_correct = False
        if targets[idx] == chr(ord("A") + max_idx):
            correct_samples += 1
            is_correct = True
        results.append((idx,
                        sum(prob_per_option),
                        np.array(prob_per_option) / sum(prob_per_option),
                        is_correct,
        ))
    print(f"--> inference_offline: accuracy = {correct_samples}/{total_samples} = {correct_samples/total_samples:.4f}")
    print(f"--> inference_offline: probability mass sum average: {np.mean(sum_probs):.4f} +/- {np.std(sum_probs):.4f}")
    return results


def run_baseline_prompting(args) -> None:
    """
    Run inference - prompting method.
    """
    with open(args.input_path, "r", encoding="utf-8") as f:
        print(f"--> run_baseline_prompting: input path = {args.input_path}")
        lines = [json.loads(line) for line in f if line.strip()]
    
    sampling_params, llm = get_llm_engine(args, mode="prompting")
    results = inference_offline(args, lines, sampling_params, llm)

    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        args.input_path.split("/")[-1].replace(".jsonl","")
        + f"_llm_inference_{args.base_model_name_or_path.replace('/','--')}.json"
    )
    with open(output_file, "w") as f:
        json.dump([
            {
                "index": res[0],
                "probability_mass": res[1],
                "probability_per_option": res[2].tolist(),
                "is_correct": res[3],
            } for res in results
        ], f, indent=4)
    print(f"--> run_baseline_prompting: saved to {output_file}")


def main():

    args = cli_args_parser()
    if args.use_logger:
        curr_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _ = start_capture(
            debug=True,
            save_path=os.path.join(ROOT_DIR, "outputs", "logs",
                                   f"llm_inference_{curr_datetime}.log"),
        )

    # check argument consistency
    assert args.input_path is not None and args.output_dir is not None, (
        "input_path and output_dir should be provided."
    )
    assert args.input_path.endswith(".jsonl"), (
        "input_path should be a .jsonl file from the scripts/preprocessing/run_prompt_formulation.py"
    )
    if args.lora_name is None and args.lora_path is not None:
        raise ValueError("LoRA name should be provided when LoRA path provided.")
    
    run_baseline_prompting(args)


if __name__ == "__main__":
    main()
