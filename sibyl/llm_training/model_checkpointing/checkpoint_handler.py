# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import json
import random
from pathlib import Path
from datetime import datetime
import torch
import time

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, cfg):
    # torch.manual_seed(103)
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_nickname
    )

    load_dir = Path.cwd() / folder_name

    if not load_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory found...skipping")
        return
    if rank == 0:
         print(f"loading model from model path: {load_dir} ")
    reader = FileSystemReader(load_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
      
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"])
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg,optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_nickname
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
            
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )
def save_fsdp_model_checkpoint_full(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")
   

    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_nickname
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_name.replace("/","--") + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)

        
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")
      


def load_model_checkpoint(model, rank, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return


    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    
    print(f"model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

   
    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_nickname
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            "optimizer" + "-" + cfg.model_nickname + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")
        print(f"Optimizer save full path = {opt_save_full_path}")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """


    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    print(f"optimizer shard loaded on rank {rank}")

def load_sharded_model_single_gpu(model,model_path):
    
    reader = FileSystemReader(model_path)
    
    state_dict = {
        "model": model.state_dict()
    }
    
    dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader= FileSystemReader(model_path),
                no_dist=True,
            )
    
    model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model

def save_peft_checkpoint(model, model_path):
    """save_pretrained peft model"""

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    
    if isinstance(model, FSDP):
        state_dict = get_model_state_dict(model, options=options)
        model.save_pretrained(model_path, state_dict=state_dict)
    else:
        model.save_pretrained(model_path)

def save_peft_checkpoint_checkpointing(
    model,
    checkpoint_path,
    optimizer = None,
    scheduler = None,
    scaler = None,
    epoch = None,
    best_val_loss = None,
    train_prep = None,
    train_loss = None,
    val_prep = None,
    val_loss = None,
    train_step_perplexity = None,
    train_step_loss = None,
    val_step_loss = None,
    val_step_perplexity = None,
    test_step_loss = None,
    test_step_perplexity = None,
    epoch_times = None,
    checkpoint_times = None,
    total_train_steps = None,
    max_steps_reached = None,
    sampler_state = None,
):

    """
    Save_pretrained peft model checkpoint and training status at the end of every epoch.
    When the training resumes, it automatically detects the existence of checkpoint directory
    and resumes from the last checkpoint.
    """

    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    if isinstance(model, FSDP):
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)
        model.save_pretrained(checkpoint_path, state_dict=state_dict)
    else:
        model.save_pretrained(checkpoint_path)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(checkpoint_path, "grad_scaler.pt"))
    if sampler_state is not None:
        torch.save(sampler_state, os.path.join(checkpoint_path, "sampler_state.pt"))

    rng_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),  # for all GPUs
        "python": random.getstate()
    }
    torch.save(rng_state, os.path.join(checkpoint_path, "rng_state.pth"))

    metadata = {
        "epoch": epoch,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "train_prep": train_prep,
        "train_loss": train_loss,
        "val_prep": val_prep,
        "val_loss": val_loss,
        "train_step_perplexity": train_step_perplexity,
        "train_step_loss": train_step_loss,
        "val_step_loss": val_step_loss,
        "val_step_perplexity": val_step_perplexity,
        "test_step_loss": test_step_loss,
        "test_step_perplexity": test_step_perplexity,
        "epoch_times": epoch_times,
        "checkpoint_times": checkpoint_times,
        "total_train_steps": total_train_steps,
        "max_steps_reached": max_steps_reached,
    }
    with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
        try:
            json.dump(metadata, f, indent=4)
        except:
            import pdb; pdb.set_trace()

    print(f"[save_peft_checkpoint]: PEFT checkpoint saved to {checkpoint_path}")

    
    
def save_model_checkpoint(model, output_dir):
    """save model when not peft and on single device"""
    
    output_file = Path(output_dir) / "model.pt"
    
    state_dict = model.state_dict()
    
    torch.save(state_dict, output_file)
    
