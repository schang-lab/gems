import argparse
import os
import math
from pathlib import Path
from functools import partial
from typing import List, Dict, Optional, Union, Tuple, Literal
from multiprocessing import Queue

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from vllm import LLM, SamplingParams


def get_llm_engine(args: argparse.Namespace,
                   mode: Literal["prompting", "agentic_cot"]) -> Tuple[SamplingParams, LLM]:
    """
    Load the LLM engine on a local device and define sampling parameters
    when mode = prompting
        only the first token probability is needed, hence sampling param is
        max_tokens=1, temp=1.0, logprobs=128
    when mode = agentic_cot
        generating the full chain-of-thought, hence sampling param is
        max_tokens ~ model max length, temp=0.0, logprobs=None
        - especially, we allocate 80% of the total length, because agentic_cot is a 2-step process
        and the reflection module (1st step) should not consume all context length.
    """
    assert mode in ["prompting", "agentic_cot"], "--> get_llm_engine(): invalid mode."
    is_prompting = mode == "prompting"
    llm = LLM(
        model=args.base_model_name_or_path,
        tensor_parallel_size=args.tp_size,
        max_logprobs=args.max_logprobs,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len if is_prompting else None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True if args.lora_path is not None else False
    )
    sampling_params = SamplingParams(
        max_tokens=1 if is_prompting else llm.llm_engine.model_config.max_model_len * 8 // 10,
        temperature=1.0 if is_prompting else 0.0,
        logprobs=128 if is_prompting else None,
    )
    return sampling_params, llm


def init_distributed(backend="nccl"):
    """
    Initialize torch.distributed, set local_rank and GPU device.
    Returns (local_rank, world_size).
    """
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def load_fsdp_model(
    model_name: str,
    local_rank: int = 0,
    lora_path: Optional[Union[str, Path]] = None,
) -> Tuple[FSDP, AutoTokenizer]:
    """
    Load a model on device and wrap it with FSDP (if distributed).
    Optionally load a LoRA module.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e8)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    if lora_path is not None:
        base_model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            torch_dtype=torch.bfloat16,
        ).to(torch.bfloat16)
    if dist.is_initialized():
        base_model = FSDP(
            base_model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank,
        )
    else:
        base_model = base_model.to(local_rank)
    base_model.eval()
    print(f"--> load_fsdp_model: device map {base_model.hf_device_map}.")
    return base_model, tokenizer


def get_embedding_fsdp(
    model_name: str,
    text: Union[str, List[str]],
    n_workers: int,
    eos_position: bool = False,
    layer_idx: Union[int, List[int]] = -1,
    batch_size: int = 1,
    lora_path: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """
    Spawns n_workers processes to load the FSDP-wrapped model in each
    and run embeddings; gathers them back on rank 0.
    """
    assert batch_size == 1, (
        "--> get_embedding_fsdp(): currently limiting batch_size = 1 "
        "due to numerical reproducibility of embeddings. "
        "For more information about batch-dependent embedding reproducibility, "
        "refer to https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/"
    )
    assert torch.cuda.device_count() >= n_workers, (
        f"--> get_embedding_fsdp(): lacking available GPUs, < {n_workers}"
    )
    if isinstance(text, str):
        text = [text]
    if isinstance(layer_idx, int):
        layer_idx = [layer_idx]

    ctx = mp.get_context("spawn")
    queue: Queue = ctx.Queue(maxsize=1)
    if n_workers == 1:
        _embedding_worker(
            0, model_name, text, eos_position, layer_idx,
            batch_size, n_workers, lora_path, queue
        )
        return queue.get()
    else:
        pc = torch.multiprocessing.spawn(
            _embedding_worker,
            args=(model_name, text, eos_position, layer_idx,
                  batch_size, n_workers, lora_path, queue),
            nprocs=n_workers,
            join=False,
        )
        embeddings: torch.Tensor = queue.get()
        pc.join()
        return embeddings


@torch.inference_mode()
def _embedding_worker(
    local_rank: int,
    model_name: str,
    text: Union[str, List[str]],
    eos_position: bool,
    layer_idx: Union[int, List[int]],
    batch_size: int,
    n_workers: int,
    lora_path: Optional[Union[str, Path]],
    queue: Queue,
) -> None:
    """
    eos_position:
        If true, extracting embedding from the EOS token. If false,
        extracting embedding from the second-to-last (i.e. just before EOS) token.
    layer_idx:
        List of layer indices to extract embeddings from.
    """
    if n_workers > 1:
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(n_workers)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        torch.cuda.set_device(local_rank)
        local_rank_, world_size_ = init_distributed()
        assert dist.is_initialized(), "--> _embedding_worker(): dist not initialized."
        assert local_rank == local_rank_, "-->_embedding_worker(): local rank mismatch"
        assert n_workers == world_size_, "-->_embedding_worker(): world size mismatch"

    model, tokenizer = load_fsdp_model(model_name, local_rank, lora_path)

    len_text = len(text)
    load_per_worker = math.ceil(len_text / n_workers)
    my_start = local_rank * load_per_worker
    my_end = (local_rank + 1) * load_per_worker
    my_workload = (text * 2)[my_start:my_end]

    embeddings_list = []
    for i in tqdm(range(0, len(my_workload), batch_size), desc="Embedding extract"):
        input = tokenizer(
            [text + tokenizer.eos_token for text in my_workload[i : i + batch_size]],
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**input, output_hidden_states=True, return_dict=True)
        hidden = torch.stack(outputs.hidden_states, dim=0).detach().cpu()
        embeddings_list.append(
            hidden[
                layer_idx, :, -1 if eos_position else -2, :
            ].transpose(0, 1)
        ) # embeddings: shape = (text index, layers, feature dimension)

    embeddings = torch.cat(embeddings_list, dim=0)

    if dist.is_initialized():
        gather_list = [None] * n_workers if local_rank == 0 else None
        dist.gather_object(embeddings, gather_list, dst=0)
        dist.barrier()
        print("--> _embedding_worker(): gathering embeddings...")
        if local_rank == 0:
            all_embeddings = torch.cat(gather_list, dim=0)[:len_text]
            queue.put(all_embeddings)
        dist.barrier()
        print("--> _embedding_worker(): all embeddings gathered.")
        dist.destroy_process_group()
        torch.cuda.empty_cache()
    else:
        queue.put(embeddings)
        del model, tokenizer
        torch.cuda.empty_cache()


def cli_args_parser():
    """
    Command line argument parser for LLM prompting / Agentic CoT baseline experiments.
    """
    parser = argparse.ArgumentParser(description="Baseline: LLM prompting.")
    parser.add_argument("--input_path", type=str,
                        default="outputs/llm_prompts/opinionqa_individual_val0p05_test0p60_evalpartial_0p40_seed42_topk_3_test.jsonl",
                        help="Input file path.")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/llm_inference",
                        help="Output directory to save the inference results.")
    parser.add_argument("--base_model_name_or_path", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="Huggingface model name or path.")
    parser.add_argument("--is_chat", action="store_true",
                        help="Whether the model is chat model. Explicitly set the flag.")
    parser.add_argument("--tp_size", type=int,
                        default=1,
                        help="Tensor parallellism.")
    parser.add_argument("--max_logprobs", type=int,
                        default=128,
                        help="Top K logprobs to be returned from vLLM engine.")
    parser.add_argument("--enable_prefix_caching", type=bool,
                        default=False,
                        help="Whether to enable prefix caching in vLLM.")
    parser.add_argument("--enforce_eager", type=bool,
                        default=True,
                        help="Whether to enforce eager mode in vLLM.")
    parser.add_argument("--max_model_len", type=int,
                        default=4096,
                        help="Could be set to AutoConfig max_position_embeddings.")
    parser.add_argument("--lora_path", type=str,
                        default=None,
                        help="LoRA on huggingface for local. If not provided, will run the base model.")
    parser.add_argument("--lora_name", type=str,
                        default=None,
                        help="Nickname of lora module for saving results.")
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=0.9,
                        help="The GPU memory utilization for vLLM.")
    parser.add_argument("--use_logger", action="store_true",
                        help="if set, will save the stdout and stderr to a file.")
    return parser.parse_args()