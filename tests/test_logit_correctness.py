# logit correctness check
# linked to sibyl/utils/llm_utils.py
# consisted of two checks: (1) against vLLM, (2) different GPU workers
# run with pytest -v tests/test_logit_correctness.py with GPU resources defined.

from typing import Optional, Union

import torch
import pytest
from vllm import LLM
from vllm.lora.request import LoRARequest

from sibyl.utils.llm_utils import get_embedding_fsdp

SAMPLE_PROMPTS = [
    "Question: When it comes to...",
    "Hello, my name is Joseph.",
    "UC Berkeley is a university located in California,!@#$%^&*()_+",
    "The future of AI is",
    "vLLM is developed by Woosuk Kwon!"
]
TOLERANCE_VLLM = 1e-2
TOLERANCE_MULTIWORKER = 1e-3
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"


def against_vllm_sanity_check(
    n_workers: int = 1,
    lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
):
    if torch.cuda.device_count() < n_workers:
        pytest.skip(
            f"Skipping test: {n_workers} workers requested, but only "
            f"{torch.cuda.device_count()} GPUs available."
        )
    result1 = get_embedding_fsdp(
        model_name=DEFAULT_MODEL,
        text=SAMPLE_PROMPTS,
        n_workers=n_workers,
        eos_position=False,
        layer_idx=-1,
    ).squeeze(1).to(torch.float32)

    vllm_model = LLM(model=DEFAULT_MODEL,
                    task="embed",
                    tensor_parallel_size=n_workers)
    vllm_output = vllm_model.encode(
        SAMPLE_PROMPTS
    ) # normalized embedding by default
    result2 = torch.vstack([
        torch.tensor(vllm_output[i].outputs.data)
        for i in range(len(vllm_output))
    ])

    result1 /= torch.norm(result1, dim=-1, keepdim=True, dtype=result1.dtype)
    assert result1.shape == result2.shape, "This should not happen."
    assert torch.allclose(result1, result2, atol=TOLERANCE_VLLM), (
        "Logit numerical instability between vLLM and FSDP."
    )
    del vllm_model
    torch.cuda.empty_cache()


def multiworker_sanity_check(
    n_workers: int = 2,
):
    if torch.cuda.device_count() < n_workers:
        pytest.skip(
            f"Skipping test: {n_workers} workers requested, but only "
            f"{torch.cuda.device_count()} GPUs available."
        )
    result1 = get_embedding_fsdp(
        model_name=DEFAULT_MODEL,
        text=SAMPLE_PROMPTS,
        n_workers=1,
        eos_position=False,
        layer_idx=-1,
    )

    result2 = get_embedding_fsdp(
        model_name=DEFAULT_MODEL,
        text=SAMPLE_PROMPTS,
        n_workers=n_workers,
        eos_position=False,
        layer_idx=-1,
    )

    assert result1.shape == result2.shape, "This should not happen."
    assert torch.allclose(result1, result2, atol=TOLERANCE_MULTIWORKER), (
        "Logit numerical instability."
    )

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 GPUs")
def test_multiworker_sanity_check():
    multiworker_sanity_check()

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires >= 1 GPU")
def test_against_vllm_sanity_check():
    against_vllm_sanity_check()

def main():
    test_multiworker_sanity_check()
    test_against_vllm_sanity_check()

if __name__ == "__main__":
    main()