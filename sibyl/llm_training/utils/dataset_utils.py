# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from sibyl.llm_training.data.concatenator import ConcatDataset
from sibyl.llm_training.datasets import DATASET_PREPROC, DATALOADER_COLLATE_FUNC
from sibyl.llm_training.utils.config_utils import get_dataloader_kwargs


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train", chat_template: bool = False
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        if split == "train":
            return dataset_config.train_split
        elif split == "valid":
            return dataset_config.valid_split
        elif split == "test":
            return dataset_config.test_split
        else:
            raise ValueError(f"Unknown split: {split}")

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
        chat_template=chat_template
    )

def get_custom_data_collator(
    dataset_processer, dataset_config
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATALOADER_COLLATE_FUNC:
        return None

    return DATALOADER_COLLATE_FUNC[dataset_config.dataset](
        dataset_processer,
        dataset_config
    )

def get_dataloader(tokenizer, dataset_config, train_config, split: str = "train"):
    dataset = get_preprocessed_dataset(tokenizer, dataset_config, split)
    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)
    
    if split == "train" and train_config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **dl_kwargs,
    )
    return dataloader
    