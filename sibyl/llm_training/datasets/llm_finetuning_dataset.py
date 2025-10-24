# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import datasets

from sibyl.utils.string_utils import prompt_chat_formatter


def get_preprocessing_dataset(
    dataset_config, tokenizer, split, chat_template, save = True,
):
    
    tokenizer.padding_side = "left"

    def _is_gpt_oss(tok):
        # either model name includes gpt-oss, or the template includes Harmony tags.
        name = (getattr(tok, "name_or_path", "") or "").lower()
        tmpl = getattr(tok, "chat_template", "") or ""
        return (
            ("gpt-oss" in name) or
            ("<|start|>" in tmpl and "<|channel|>" in tmpl and "<|message|>" in tmpl)
        )
    _uses_harmony = _is_gpt_oss(tokenizer)
    
    chat_template = True if chat_template.lower() == 'true' else False
    print(f"--> preprocessing dataset: is_chat_template = {chat_template}")

    def tokenize_add_label(sample):

        if not chat_template: # using pretrained base model
            prompt = tokenizer.encode(
                tokenizer.bos_token + sample["prompt"],
                add_special_tokens=False
            )
            answer: int = tokenizer.encode(
                "Answer: " + sample["label"],
                add_special_tokens=False
            )[-1]
        else: # using chat model
            messages = prompt_chat_formatter(sample["prompt"])
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=True,
                add_generation_prompt=True,
            )
            if _uses_harmony: # gpt-oss special token handling
                assistant_header = tokenizer.encode(
                    "<|channel|>final<|message|>", add_special_tokens=False
                )
                prompt = prompt + assistant_header
                answer: int = tokenizer.encode(
                    sample["label"], add_special_tokens=False
                )[0]
            else:
                answer: int = tokenizer.encode(
                    "Answer: " + sample["label"],
                    add_special_tokens=False
                )[-1]

        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "target_token_id": answer,
            }
        return sample
        
    dataset = datasets.load_dataset(
        'json', data_files=split
    )['train']
    dataset = dataset.map(
        tokenize_add_label,
        remove_columns=list(dataset.features),
        num_proc=32
    )
    return dataset