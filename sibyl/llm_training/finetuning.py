# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import json
import dataclasses
import random
from collections import Counter
from datetime import datetime

import torch.distributed
import fire
import torch
import torch.optim as optim
from peft import get_peft_model, PeftModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR, LambdaLR
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoProcessor, 
    LlamaForCausalLM,
    MistralForCausalLM, # mistral support (custom)
    Qwen2ForCausalLM, 
    Qwen3ForCausalLM,
    GptOssForCausalLM,
    MllamaForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer # mistral support (custom)
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer
from transformers.models.mllama.modeling_mllama import  MllamaSelfAttentionDecoderLayer,MllamaCrossAttentionDecoderLayer,MllamaVisionEncoderLayer

from sibyl.llm_training.configs import fsdp_config as FSDP_CONFIG
from sibyl.llm_training.configs import train_config as TRAIN_CONFIG
from sibyl.llm_training.configs import quantization_config  as QUANTIZATION_CONFIG
from sibyl.llm_training.data.concatenator import ConcatDataset
from sibyl.llm_training.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from sibyl.llm_training.utils import fsdp_auto_wrap_policy
from sibyl.llm_training.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
    check_fsdp_config,
)
from sibyl.llm_training.utils.dataset_utils import get_preprocessed_dataset,get_custom_data_collator

from sibyl.llm_training.utils.fsdp_utils import hsdp_device_mesh
from sibyl.llm_training.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from accelerate.utils import is_xpu_available
from warnings import warn, simplefilter

simplefilter(action='ignore', category=FutureWarning)


def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from sibyl.llm_training.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run


def lr_lambda(current_step, warmup_steps, total_steps):
    """
    Cosine scheduler with warmup.
    Args:
        current_step: current step in the training loop
        warmup_steps: number of steps for warmup
        total_steps: total number of steps for training
    Returns:
        float: learning rate multiplier
    """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress) * 3.14159265358979323846)))


def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    if not hasattr(train_config, "checkpoint_interval"):
        train_config.checkpoint_interval = 1 # save checkpoint evey epoch
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
            formatted_time_tensor = torch.tensor([ord(c) for c in formatted_time], dtype=torch.int32, device="cuda")
        else:
            formatted_time_tensor = torch.empty(15, dtype=torch.int32, device="cuda")
        # synchronize the formatted time tensor across all ranks to ensure the output directory is consistent
        torch.distributed.barrier()
        torch.distributed.broadcast(formatted_time_tensor, src=0)
        torch.distributed.barrier()        
        formatted_time = ''.join([chr(c) for c in formatted_time_tensor.cpu().tolist() if c != 0])
    else:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")    
    train_config.output_dir += formatted_time
    print(f"--> Output Directory (Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'}): {train_config.output_dir}")

    wandb_run = None
    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank==0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

    #setting quantization configs
    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn("Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.", FutureWarning)
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError("8bit quantization is not supported with FSDP, please use 4bit quantization")

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    config = AutoConfig.from_pretrained(train_config.model_name)
    if config.model_type == "mllama":
        is_vision = True
        model = MllamaForConditionalGeneration.from_pretrained(
        train_config.model_name,
        quantization_config=bnb_config,
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
        torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
    )
        processor = AutoProcessor.from_pretrained(train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name)
        processor.tokenizer.padding_side='right'
        model.supports_gradient_checkpointing = True
        model.language_model.supports_gradient_checkpointing = True
    elif config.model_type == "llama":
        is_vision = False
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = LlamaForCausalLM.from_pretrained(
                    train_config.model_name,
                    quantization_config=bnb_config,
                    use_cache=use_cache,
                    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                    device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                    torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
                )
            else:
                llama_config = AutoConfig.from_pretrained(train_config.model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)
        else:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
            )
    elif config.model_type == "mistral": # mistral support (custom)
        is_vision = False
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = MistralForCausalLM.from_pretrained(
                    train_config.model_name,
                    quantization_config=bnb_config,
                    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                    device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                    torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
                )
            else:
                mistral_config = AutoConfig.from_pretrained(train_config.model_name)
                mistral_config.use_cache = use_cache
                with torch.device("meta"):
                    model = MistralForCausalLM(mistral_config)
        else:
            model = MistralForCausalLM.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
            )
    elif config.model_type == "qwen3":  # qwen3 support (custom)
        is_vision = False
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = Qwen3ForCausalLM.from_pretrained(
                    train_config.model_name,
                    quantization_config=bnb_config,
                    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                    device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                    torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
                )
            else:
                qwen_cfg = AutoConfig.from_pretrained(train_config.model_name)
                qwen_cfg.use_cache = use_cache
                with torch.device("meta"):
                    model = Qwen3ForCausalLM(qwen_cfg)
        else:
            model = Qwen3ForCausalLM.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
            )
    elif config.model_type == "qwen2":  # optional: qwen2 support
        is_vision = False
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = Qwen2ForCausalLM.from_pretrained(
                    train_config.model_name,
                    quantization_config=bnb_config,
                    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                    device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                    torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
                )
            else:
                qwen2_cfg = AutoConfig.from_pretrained(train_config.model_name)
                qwen2_cfg.use_cache = use_cache
                with torch.device("meta"):
                    model = Qwen2ForCausalLM(qwen2_cfg)
        else:
            model = Qwen2ForCausalLM.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
            )
    elif config.model_type == "gpt_oss":  # gpt-oss support (custom)
        is_vision = False
        # Note: GPT-OSS weights ship with MXFP4 kernels via `kernels` package; we do NOT
        # pass BitsAndBytes `quantization_config` here to avoid conflicts with MXFP4.
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = GptOssForCausalLM.from_pretrained(
                    train_config.model_name,
                    device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                    torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
                )
            else:
                gptoss_cfg = AutoConfig.from_pretrained(train_config.model_name)
                gptoss_cfg.use_cache = use_cache
                with torch.device("meta"):
                    model = GptOssForCausalLM(gptoss_cfg)
        else:
            model = GptOssForCausalLM.from_pretrained(
                train_config.model_name,
                device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
            )
    else:
        raise ValueError(f"Model type {config.model_type} is not supported. Please use llama or mllama model.")
    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name)
    if not tokenizer.pad_token_id: 
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16 and not train_config.quantization:
        model.to(torch.bfloat16)
        
    if train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        # if train_config.from_peft_checkpoint:
        #     model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
        #     peft_config = model.peft_config['default']
        checkpoint_path = os.path.dirname(train_config.output_dir)
        checkpoint_path = os.path.join(checkpoint_path, "peft_checkpointing")
        if os.path.exists(checkpoint_path):
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
            peft_config = model.peft_config['default']
        else: # Generate the peft config and start fine-tuning from original model 
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        if wandb_run:
            wandb_run.config.update(peft_config)
        model.print_trainable_parameters()

    hsdp_device_mesh_plan = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh_plan = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        check_fsdp_config(fsdp_config)
        
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        # Create the FSDP wrapper for MllamaSelfAttentionDecoderLayer,MllamaSelfAttentionDecoderLayer,MllamaVisionEncoderLayer in vision models
        if is_vision:
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [MllamaSelfAttentionDecoderLayer,MllamaSelfAttentionDecoderLayer,MllamaVisionEncoderLayer])
        else:
        # Create the FSDP wrapper for LlamaDecoderLayer in text models
            if config.model_type == "mistral": # mistral support (custom)
                my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [MistralDecoderLayer])
            elif config.model_type == "qwen3":
                my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [Qwen3DecoderLayer])
            elif config.model_type == "qwen2":
                my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [Qwen2DecoderLayer])
            elif config.model_type == "gpt_oss":
                my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [GptOssDecoderLayer])
            else:
                my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer])
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:            
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            apply_fsdp_checkpointing(model)                      
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")
    dataset_config = generate_dataset_config(train_config, kwargs)
    if is_vision:
        dataset_processer = processor
    else:
        dataset_processer = tokenizer

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="train",
        chat_template=train_config.is_chat,
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="valid",
        chat_template=train_config.is_chat,
    )
    print(f"--> Validation Set Length = {len(dataset_val)}")

    dataset_test = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="test",
        chat_template=train_config.is_chat,
    )
    print(f"--> Test Set Length = {len(dataset_test)}")

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        if is_vision:
            raise ValueError("Packing is not supported for vision datasets")
        else:
            dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, dataset_processer, "train")
    print("length of dataset_train", len(dataset_train))
    custom_data_collator = get_custom_data_collator(dataset_processer,dataset_config)
    if custom_data_collator:
        print("custom_data_collator is used")
        train_dl_kwargs["collate_fn"] = custom_data_collator
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            if is_vision:
                raise ValueError("Packing is not supported for vision datasets")
            else:
                dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, dataset_processer, "val")
        if custom_data_collator:
            val_dl_kwargs["collate_fn"] = custom_data_collator

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
        if len(eval_dataloader) == 0:
            raise ValueError("The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set.")
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    test_dataloader = None
    if train_config.run_test:
        test_dl_kwargs = get_dataloader_kwargs(train_config, dataset_test, dataset_processer, "test")
        if custom_data_collator:
            test_dl_kwargs["collate_fn"] = custom_data_collator

        test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **test_dl_kwargs,
        )
        print(f"--> Num of Test Set Batches loaded = {len(test_dataloader)}")
        if len(test_dataloader) == 0:
            raise ValueError("The test set size is too small for dataloader to load even one batch. Please increase the size of test set.")
        else:
            print(f"--> Num of Test Set Batches loaded = {len(test_dataloader)}")

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr= train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    if train_config.which_scheduler == "cosine":
        total_steps = int(
            len(train_dataloader)
            * train_config.num_epochs / train_config.gradient_accumulation_steps
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_lambda(
                current_step = step,
                warmup_steps = int(train_config.warmup_ratio * float(total_steps)),
                total_steps = total_steps,
            )
        )
        print(f"--> Using Cosine Scheduler with total_steps = {total_steps}, warmup_steps = {int(train_config.warmup_ratio * total_steps)}")
    elif train_config.which_scheduler == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=1,
            gamma=train_config.gamma ** (
                1.0 / float(len(train_dataloader))
                * float(train_config.gradient_accumulation_steps)
            ),
        )
        print(f"--> Using Step Scheduler with gamma = {scheduler.gamma}")
    
    # load checkpoint
    scaler_dict = None
    # sampler_state = None
    starting_epoch = 0
    best_val_loss = torch.tensor(float("inf")).to(torch.device("cuda"))
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []
    train_step_perplexity = []
    train_step_loss = []
    val_step_loss = []
    val_step_perplexity = []
    test_step_loss = []
    test_step_perplexity = []
    epoch_times = []
    checkpoint_times = []
    total_train_steps = 0
    max_steps_reached = False
    results = {}

    if train_config.use_peft:
        checkpoint_path = os.path.dirname(train_config.output_dir)
        checkpoint_path = os.path.join(checkpoint_path, "peft_checkpointing")
        if os.path.exists(checkpoint_path):
            if (
                os.path.exists(os.path.join(checkpoint_path, "optimizer.pt"))
                and os.path.exists(os.path.join(checkpoint_path, "scheduler.pt"))
                and os.path.exists(os.path.join(checkpoint_path, "grad_scaler.pt"))
                and os.path.exists(os.path.join(checkpoint_path, "metadata.json"))
            ):
                optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scheduler.pt")))
                scaler_dict = torch.load(os.path.join(checkpoint_path, "grad_scaler.pt"))
                print("Loaded optimizer, scheduler and scaler from checkpoint")
                with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
                    metadata = json.load(f)
                    starting_epoch = metadata["epoch"]
                    best_val_loss = torch.tensor(metadata["best_val_loss"])
                    best_val_loss = best_val_loss.to(torch.device("cuda"))
                    train_prep = metadata["train_prep"]
                    train_loss = metadata["train_loss"]
                    val_prep = metadata["val_prep"]
                    val_loss = metadata["val_loss"]
                    train_step_perplexity = metadata["train_step_perplexity"]
                    train_step_loss = metadata["train_step_loss"]
                    val_step_loss = metadata["val_step_loss"]
                    val_step_perplexity = metadata["val_step_perplexity"]
                    test_step_loss = metadata["test_step_loss"]
                    test_step_perplexity = metadata["test_step_perplexity"]
                    epoch_times = metadata["epoch_times"]
                    checkpoint_times = metadata["checkpoint_times"]
                    total_train_steps = metadata["total_train_steps"]
                    max_steps_reached = metadata["max_steps_reached"]
                    # if train_dataloader.sampler has a set_epoch method (FSDP case), set the epoch
                    if hasattr(train_dataloader.sampler, "set_epoch"):
                        train_dataloader.sampler.set_epoch(starting_epoch)
                rng_state = torch.load(os.path.join(checkpoint_path, "rng_state.pth"))
                torch.set_rng_state(rng_state['torch'])
                torch.cuda.set_rng_state_all(rng_state['cuda'])
                random.setstate(rng_state['python'])
    results = train(
        model = model,
        train_dataloader = train_dataloader,
        eval_dataloader = eval_dataloader,
        test_dataloader = test_dataloader,
        tokenizer = tokenizer,
        optimizer = optimizer,
        lr_scheduler = scheduler,
        gradient_accumulation_steps = train_config.gradient_accumulation_steps,
        train_config = train_config,
        fsdp_config = fsdp_config if train_config.enable_fsdp else None,
        local_rank = local_rank if train_config.enable_fsdp else None,
        rank = rank if train_config.enable_fsdp else None,
        wandb_run = wandb_run,
        scaler_dict = scaler_dict,
        train_prep = train_prep,
        train_loss = train_loss,
        val_prep = val_prep,
        val_loss = val_loss,
        train_step_perplexity = train_step_perplexity,
        train_step_loss = train_step_loss,
        val_step_loss = val_step_loss,
        val_step_perplexity = val_step_perplexity,
        test_step_loss = test_step_loss,
        test_step_perplexity = test_step_perplexity,
        epoch_times = epoch_times,
        checkpoint_times = checkpoint_times,
        results = results,
        best_val_loss = best_val_loss,
        total_train_steps = total_train_steps,
        max_steps_reached = max_steps_reached,
        starting_epoch = starting_epoch,

    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
        if train_config.use_wandb:
            for k,v in results.items():
                wandb_run.summary[k] = v

if __name__ == "__main__":
    fire.Fire(main)
